FROM python:3.11-slim

# Hugging Face Spaces ke liye non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# CPU wala PyTorch (chhota aur fast install)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# Backend (8000) aur frontend (7860) dono ek saath chalao
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run medibot.py --server.port 7860 --server.address 0.0.0.0"]