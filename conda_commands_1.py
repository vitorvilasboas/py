
	sudo chmod +x Anaconda3-XXXX.XX-Linux-x86_64.sh # permissão para execução
	
	./Anaconda3-XXXX.XX-Linux-x86_64.sh # instalar conda
	
	source ~/.bashrc # add ao bash	
	
	conda update -n base -c defaults conda #atualizar conda
	
	conda create --name novoAmbiente python=3.7 # criar novo ambiente
	
	conda env list # listar ambientes
	
	conda activate novoAmbiente # ativar ambiente
	
	conda list # listar pacotes
	
	conda install numpy # instalar pacotes via conda
	
	conda install scipy
	
	conda install matplotlib
	
	pip install scipy # instalar pacotes com pip
	
	conda deactivate # desativar ambiente