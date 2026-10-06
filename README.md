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

Certifique se de carregar a versão do qwen com a quantidade correspondente de parâmetros, neste caso, estamos com o ```qwen2.5:3b```. Versões diferentes no serve podem causar problemas

Execute:

```js
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 3 - Na raiz do projeto:

Execute:

```js
docker compose build --no-cache

```
```js
docker compose up
```

### 4 - Acesse:

[localhost:8000](http://localhost:8000/)
