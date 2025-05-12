# Makefile

# Definir las rutas de los archivos
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python

# Instalar dependencias
install:
	@echo "🔧 Instalando dependencias..."
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Entrenar el modelo
train:
	@echo "🚀 Iniciando el entrenamiento..."
	$(PYTHON) src/train.py

# Ejecutar pruebas básicas (por ahora, solo un placeholder)
test:
	@echo "🧪 Ejecutando pruebas..."
	pytest tests/

# Limpiar (eliminar el entorno virtual)
clean:
	@echo "🧹 Limpiando el proyecto..."
	rm -rf $(VENV_DIR)