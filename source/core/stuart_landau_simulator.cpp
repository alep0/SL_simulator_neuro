/**
 * @file stuart_landau_simulator.cpp
 * @brief C++ accelerated Stuart-Landau neural network simulator with pybind11 bindings.
 *
 * Implements a delayed-coupled Stuart-Landau oscillator network on a structural
 * connectome. Supports two connectivity layers (C1/C2), additive Gaussian noise,
 * and OpenMP parallelism for the coupling and history-shift loops.
 *
 * Build via setup.py:
 *   python setup.py build_ext --inplace
 *
 * Python interface (module name: stuart_landau_simulator):
 *   sim = StuartLandauSimulator(N, max_history, dt, dt_save, tmax, t_prev,
 *                               sig_noise, mean_delay)
 *   sim.set_connectivity(C1, Delays1 [, C2, Delays2])
 *   trajectory = sim.simulate(K, a, f)   # returns np.ndarray (N, n_save)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include <complex>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Helper: format a log message with a consistent prefix
// ---------------------------------------------------------------------------
static void log_info(const std::string& msg) {
    std::cout << "[StuartLandau] " << msg << std::endl;
}

// ---------------------------------------------------------------------------
class StuartLandauSimulator {
private:
    // Network dimensions
    int N_;
    int max_history_;

    // Time parameters
    double dt_;
    double dt_save_;
    double tmax_;
    double t_prev_;

    // Physical parameters
    double sig_noise_;
    double mean_delay_;   // formerly MD; controls whether history buffer shifts

    // Connectivity matrices (two optional layers)
    std::vector<std::vector<double>> C1_;
    std::vector<std::vector<double>> C2_;
    std::vector<std::vector<int>>    Delays1_;
    std::vector<std::vector<int>>    Delays2_;
    bool use_C2_;

    // Random number generation
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_;

    // -----------------------------------------------------------------------
    // Internal: copy a 2-D numpy array (double) into a vector-of-vectors
    // -----------------------------------------------------------------------
    void copy_double_matrix(py::array_t<double>& arr,
                            std::vector<std::vector<double>>& dst) {
        auto buf = arr.request();
        auto* ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                dst[i][j] = ptr[i * N_ + j];
    }

    // -----------------------------------------------------------------------
    // Internal: copy a 2-D numpy array (int) into a vector-of-vectors
    // -----------------------------------------------------------------------
    void copy_int_matrix(py::array_t<int>& arr,
                         std::vector<std::vector<int>>& dst) {
        auto buf = arr.request();
        auto* ptr = static_cast<int*>(buf.ptr);
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                dst[i][j] = ptr[i * N_ + j];
    }

public:
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    StuartLandauSimulator(int    N,
                          int    max_history,
                          double dt,
                          double dt_save,
                          double tmax,
                          double t_prev,
                          double sig_noise,
                          double mean_delay)
        : N_(N),
          max_history_(max_history),
          dt_(dt),
          dt_save_(dt_save),
          tmax_(tmax),
          t_prev_(t_prev),
          sig_noise_(sig_noise),
          mean_delay_(mean_delay),
          use_C2_(false),
          rng_(std::random_device{}()),
          normal_dist_(0.0, 1.0)
    {
        if (N_ <= 0)
            throw std::invalid_argument("N must be positive.");
        if (max_history_ <= 0)
            throw std::invalid_argument("max_history must be positive.");
        if (dt_ <= 0.0)
            throw std::invalid_argument("dt must be positive.");
        if (dt_save_ < dt_)
            throw std::invalid_argument("dt_save must be >= dt.");
        if (tmax_ <= 0.0)
            throw std::invalid_argument("tmax must be positive.");

        C1_.assign(N_, std::vector<double>(N_, 0.0));
        Delays1_.assign(N_, std::vector<int>(N_, 0));

        log_info("Simulator constructed. N=" + std::to_string(N_) +
                 ", max_history=" + std::to_string(max_history_) +
                 ", dt=" + std::to_string(dt_) +
                 ", tmax=" + std::to_string(tmax_));
    }

    // -----------------------------------------------------------------------
    // Set connectivity (C1 mandatory; C2 optional)
    // -----------------------------------------------------------------------
    void set_connectivity(py::array_t<double> C1_arr,
                          py::array_t<int>    Delays1_arr,
                          py::object          C2_obj     = py::none(),
                          py::object          Delays2_obj = py::none())
    {
        copy_double_matrix(C1_arr,    C1_);
        copy_int_matrix   (Delays1_arr, Delays1_);

        if (!C2_obj.is_none()) {
            if (Delays2_obj.is_none())
                throw std::invalid_argument(
                    "Delays2 must be provided when C2 is provided.");

            use_C2_ = true;
            C2_.assign(N_, std::vector<double>(N_, 0.0));
            Delays2_.assign(N_, std::vector<int>(N_, 0));

            auto C2_arr     = C2_obj.cast<py::array_t<double>>();
            auto Delays2_arr = Delays2_obj.cast<py::array_t<int>>();
            copy_double_matrix(C2_arr,     C2_);
            copy_int_matrix   (Delays2_arr, Delays2_);

            log_info("Two-layer connectivity loaded (C1 + C2).");
        } else {
            log_info("Single-layer connectivity loaded (C1 only).");
        }
    }

    // -----------------------------------------------------------------------
    // Main simulation
    // -----------------------------------------------------------------------
    py::array_t<double> simulate(double              K,
                                 double              a,
                                 py::array_t<double> f_arr)
    {
        // Validate frequency array length
        auto f_buf = f_arr.request();
        if (f_buf.shape[0] != N_)
            throw std::invalid_argument(
                "f must have length N=" + std::to_string(N_) +
                ", got " + std::to_string(f_buf.shape[0]));
        double* f_ptr = static_cast<double*>(f_buf.ptr);

        const int n_steps       = static_cast<int>((tmax_ + t_prev_) / dt_);
        const int n_save        = static_cast<int>(tmax_ / dt_save_);
        const int save_interval = static_cast<int>(dt_save_ / dt_);
        const int log_interval  = std::max(1, n_steps / 10);

        log_info("Simulation starting: n_steps=" + std::to_string(n_steps) +
                 ", n_save=" + std::to_string(n_save) +
                 ", K=" + std::to_string(K) +
                 ", a=" + std::to_string(a));

        // ------------------------------------------------------------------
        // Pre-compute constants
        // ------------------------------------------------------------------
        std::vector<Complex>              iomega(N_);
        std::vector<std::vector<double>>  kC1(N_, std::vector<double>(N_));
        std::vector<std::vector<double>>  kC2;

        if (use_C2_)
            kC2.assign(N_, std::vector<double>(N_, 0.0));

        for (int i = 0; i < N_; ++i) {
            iomega[i] = Complex(0.0, 2.0 * M_PI * f_ptr[i]);
            for (int j = 0; j < N_; ++j) {
                kC1[i][j] = K * C1_[i][j] * dt_;
                if (use_C2_)
                    kC2[i][j] = K * C2_[i][j] * dt_;
            }
        }

        const double  dsig      = std::sqrt(dt_) * sig_noise_;
        const Complex a_complex(a, 0.0);

        // ------------------------------------------------------------------
        // Initialise state history with small noise
        // ------------------------------------------------------------------
        std::vector<std::vector<Complex>> Z(N_,
            std::vector<Complex>(max_history_));
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < max_history_; ++j)
                Z[i][j] = Complex(dt_ * normal_dist_(rng_),
                                  dt_ * normal_dist_(rng_));

        // ------------------------------------------------------------------
        // Allocate output array
        // ------------------------------------------------------------------
        auto result     = py::array_t<double>({N_, n_save});
        auto result_buf = result.request();
        double* out_ptr = static_cast<double*>(result_buf.ptr);

        // Temporary per-step vectors
        std::vector<Complex> Znow(N_), dz(N_), noise(N_),
                             coupling1(N_), coupling2(N_);

        int    nt = 0;
        double t  = dt_;

        // ------------------------------------------------------------------
        // Main loop
        // ------------------------------------------------------------------
        for (int step = 0; step < n_steps; ++step, t += dt_) {

            if (step % log_interval == 0) {
                double pct = 100.0 * t / tmax_;
                log_info("Progress: " + std::to_string(pct) +
                         "% (t=" + std::to_string(t) + "s)");
            }

            // Snapshot current state
            for (int i = 0; i < N_; ++i)
                Znow[i] = Z[i][max_history_ - 1];

            // Stuart-Landau local dynamics
#pragma omp parallel for schedule(static)
            for (int i = 0; i < N_; ++i) {
                double abs_sq = std::norm(Znow[i]);
                dz[i] = Znow[i] * (a_complex + iomega[i] - abs_sq) * dt_;
            }

            // Delayed coupling
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N_; ++n) {
                Complex s1(0.0, 0.0), s2(0.0, 0.0);
                for (int j = 0; j < N_; ++j) {
                    if (kC1[n][j] != 0.0) {
                        int di = Delays1_[n][j] - 1;
                        s1 += kC1[n][j] * (Z[j][di] - Znow[n]);
                    }
                    if (use_C2_ && kC2[n][j] != 0.0) {
                        int di = Delays2_[n][j] - 1;
                        s2 += kC2[n][j] * (Z[j][di] - Znow[n]);
                    }
                }
                coupling1[n] = s1;
                coupling2[n] = s2;
            }

            // Shift history buffer
            if (mean_delay_ > 0.0) {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < N_; ++i)
                    for (int j = 0; j < max_history_ - 1; ++j)
                        Z[i][j] = Z[i][j + 1];
            }

            // Add noise and update state
            for (int i = 0; i < N_; ++i) {
                noise[i] = Complex(dsig * normal_dist_(rng_),
                                   dsig * normal_dist_(rng_));
                Complex coupling_total = coupling1[i];
                if (use_C2_)
                    coupling_total += coupling2[i];
                Z[i][max_history_ - 1] =
                    Znow[i] + dz[i] + noise[i] + coupling_total;
            }

            // Record output
            if (t > t_prev_ && (step % save_interval == 0) && (nt < n_save)) {
                for (int i = 0; i < N_; ++i)
                    out_ptr[i * n_save + nt] = Z[i][max_history_ - 1].real();
                ++nt;
            }
        }

        log_info("Simulation complete. Saved " + std::to_string(nt) +
                 " time points.");
        return result;
    }
};

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(stuart_landau_simulator, m) {
    m.doc() = R"pbdoc(
        C++ accelerated Stuart-Landau neural network simulator.

        Provides ``StuartLandauSimulator``, a delayed-coupled oscillator
        network that models resting-state neural dynamics on an empirical
        structural connectome.
    )pbdoc";

    py::class_<StuartLandauSimulator>(m, "StuartLandauSimulator",
        R"pbdoc(
        Delayed-coupled Stuart-Landau oscillator network.

        Parameters
        ----------
        N : int
            Number of oscillators (network nodes).
        max_history : int
            History buffer length in time-steps (>= max delay + 1).
        dt : float
            Integration time step (s).
        dt_save : float
            Output sampling interval (s). Must satisfy dt_save >= dt.
        tmax : float
            Total simulation time (s).
        t_prev : float
            Warm-up time before recording begins (s).
        sig_noise : float
            Amplitude of complex Gaussian noise.
        mean_delay : float
            Mean axonal delay (s). If 0 the history buffer is not shifted.
        )pbdoc")

        .def(py::init<int, int, double, double, double, double, double, double>(),
             py::arg("N"),
             py::arg("max_history"),
             py::arg("dt"),
             py::arg("dt_save"),
             py::arg("tmax"),
             py::arg("t_prev"),
             py::arg("sig_noise"),
             py::arg("mean_delay"))

        .def("set_connectivity",
             &StuartLandauSimulator::set_connectivity,
             py::arg("C1"),
             py::arg("Delays1"),
             py::arg("C2")      = py::none(),
             py::arg("Delays2") = py::none(),
             R"pbdoc(
             Load connectivity and delay matrices.

             Parameters
             ----------
             C1 : np.ndarray, shape (N, N), dtype float64
                 Primary normalised weight matrix.
             Delays1 : np.ndarray, shape (N, N), dtype int32
                 Delay indices for C1 (in time-steps).
             C2 : np.ndarray or None
                 Optional secondary weight matrix.
             Delays2 : np.ndarray or None
                 Delay indices for C2. Required when C2 is given.
             )pbdoc")

        .def("simulate",
             &StuartLandauSimulator::simulate,
             py::arg("K"),
             py::arg("a"),
             py::arg("f"),
             R"pbdoc(
             Run the simulation and return the real-part trajectory.

             Parameters
             ----------
             K : float
                 Global coupling strength.
             a : float
                 Bifurcation parameter (negative = damped, positive = limit cycle).
             f : np.ndarray, shape (N,), dtype float64
                 Natural oscillation frequencies (Hz) for each node.

             Returns
             -------
             trajectory : np.ndarray, shape (N, n_save), dtype float64
                 Real part of the complex oscillator state recorded every dt_save.
             )pbdoc");
}
