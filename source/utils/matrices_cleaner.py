"""
Programa para limpiar las matrices de data de nan

v0:
    - Carga, limpia, guarda, envoltura.
    
"""

import numpy as np
from pathlib import Path

def matrix_2_file(matriz, nombre_archivo ):
    # Save W and D matrices
	try:
		with open(nombre_archivo, 'w') as archivo:
			for fila in matriz:
				fila_texto = ' '.join(map(str, fila))  # Convertir elementos de la fila a texto y unirlos con espacios
				archivo.write(fila_texto + '\n')       # Escribir la fila en el archivo, seguida de un salto de línea
	except FileNotFoundError:
		print("File not found: " + nombre_archivo)
	return

def file_2_matrix(dir_dat: str, file_name: str) -> np.ndarray:
    """Load matrix from text file - Ubuntu compatible"""
    file_path = f"{dir_dat}/{file_name}"
    #print(file_path)
    try:
        matrix = np.loadtxt(str(file_path))
        return matrix
    except Exception as e:
        raise FileNotFoundError(f'Failed to open file: {file_path}. Error: {e}')
        
def clean_matrix( matrix ):        
    
    N = len( matrix )
    clean_matrix = np.zeros( ( N, N ) )
    
    for i in range( N ):
        for j in range( N ):
            if( j > i and np.isnan( matrix[i,j] ) == False ):
                clean_matrix[i,j] = matrix[i,j]
                                
    return clean_matrix

def clean_data( rats, root_path, filter_name, group_names, var_names ):
    
    for group_name in group_names:
        for rat in rats:
            for var_name in var_names:
                print( f"group_name: {group_name}, rat: {rat}, var_name: {var_name} processed." )
                data_raw_dir = Path( root_path ) / "Data" / "raw" / group_name / "FA_RN_SI_v0-1_th-0.0" / filter_name / rat
                output_dir = Path( root_path ) / "Data" / "processed" / group_name / "FA_RN_SI_v0-1_th-0.0" / filter_name / rat
        
                output_dir.mkdir( parents=True, exist_ok=True )
            
                matrix = file_2_matrix( data_raw_dir, f"th-0.0_{rat}_{var_name}.txt" )        
                matrix_2_file( clean_matrix( matrix ), output_dir / f"th-0.0_{rat}_{var_name}.txt" )
                
    print( f"Cleaned from: {data_raw_dir}" )
    print( f"Saved to: {output_dir}" )
    
if __name__ == "__main__":
    
    group_names = [ "t1", "t2" ]
    rats = [ "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", 
             "R10", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19" ]
    var_names = [ "w", "d", "v", "tau", "fa" ]
    
    root_path = "/mnt/c/Users/aleph/Desktop/Job/Code/GitHub/Tests/SL_simulator_neuro"    
    filter_name = "filter_kick_out"
    
    clean_data( rats, root_path, filter_name, group_names, var_names )
    