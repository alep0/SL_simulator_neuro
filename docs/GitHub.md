cd path/to/my-project

git init
git branch -M main
git add .
git status

git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git remote -v
git push -u origin main

# Crear claves
ssh-keygen -t ed25519 -C "tu_email@ejemplo.com"
cat ~/.ssh/id_ed25519.pub

# Si ya hay claves
ls -l ~/.ssh

git remote set-url origin git@github.com:alep0/SL_simulator_neuro.git

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh -T git@github.com
ssh-add -l
git clone git@github.com:alep0/SL_simulator_neuro.git
