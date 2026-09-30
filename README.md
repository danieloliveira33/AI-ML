# AI-ML - Como excutar:

### 1 - Baixar o Ollama

Linux / Mac / WSL2:

```js
curl -fsSL https://ollama.com/install.sh | sh
```

Windows (Powershell):

```js
irm https://ollama.com/install.ps1 | iex
```

### 2 - Iniciar o Ollama

Execute:

```js
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 3 - Na raiz do projeto:

Execute:

```js
docker compose up
```

### 4 - Acesse:

[localhost:8000](localhost:8000)
