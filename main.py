import whisper
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate

def transcribe_audio(audio_path):
  """
  Transcreve o áudio utilizando o modelo Whisper.
  
  :param audio_path: Caminho do arquivo de áudio
  :return: Texto transcrito
  """
  print("🔊 Iniciando transcrição...")
  model = whisper.load_model("base")
  result = model.transcribe(audio_path)
  print("✅ Transcrição concluída!\n")
  return result['text']

def summarize_with_ollama(text):
  """
  Utiliza LangChain com Ollama para gerar resumo com pontos-chave.
  
  :param text: Texto a ser resumido
  :return: Resumo/pontos-chave
  """
  print("🧠 Enviando transcrição para Ollama via LangChain...")
  
  # Inicializa o modelo Ollama com o modelo local
  llm = OllamaLLM(model="llama3.1:8b")
  
  # Template para gerar pontos-chave
  template = """
  Resuma o seguinte texto em uma lista de pontos-chave importantes, claros e objetivos:
  
  {text}
  
  Resuma agora:
  """
  prompt = PromptTemplate(template=template, input_variables=["text"])
  
  result = prompt | llm

  summary = result.invoke({"text": text})
  
  print("✅ Resumo concluído!\n")
  return summary

def main():
  audio_file = "exemplo_audio.ogg"  # Caminho do arquivo de áudio
  
  # Passo 1: Transcrição
  transcript = transcribe_audio(audio_file)
  print("📝 Texto Transcrito:\n")
  print(transcript)
  
  # Passo 2: Resumo com pontos-chave
  summary = summarize_with_ollama(transcript)
  print("📌 Pontos-Chave:\n")
  print(summary)

if __name__ == "__main__":
  main()
