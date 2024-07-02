# Käytä NVIDIAn CUDA 12.4 base-konttia
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Kopioi lähdekoodi konttiin
WORKDIR /workspace
COPY . .

# Asenna Python-paketit
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio gymnasium[classic-control] stable-baselines3 moviepy optuna

# Määritä CUDA- ja cuDNN-kirjastojen polut
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

# Tarkista, että kirjastot ovat oikeilla paikoillaan
RUN ldconfig

