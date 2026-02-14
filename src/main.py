import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

rutas = [
    "Documentos\\receta_pastel_1.pdf", 
    "Documentos\\receta_pastel_2.pdf",
    "Documentos\\receta_pastel_3.pdf",
    "Documentos\\receta_pastel_4.pdf"
    ]


def extraer_texto(ruta_pdf):
    for ruta in rutas:
        with open(ruta_pdf, 'rb') as archivo:
            lector_pdf = PyPDF2.PdfReader(archivo)
            texto = ""
            for page in range(0, len(lector_pdf.pages)):
                texto += lector_pdf.pages[page].extract_text()
    return texto

#print(extraer_texto(rutas[0]))

def crear_chunks(text, chunk_size=100, overlap=50):
    palabras = text.split(" ")
    chunks = []
    for i in range(0, len(palabras), chunk_size - overlap):
        chunk = ' '.join(palabras[i : i+chunk_size])
        chunks.append(chunk)
    return chunks

#text = extraer_texto(rutas[0])
#chunks = crear_chunks(text)
#print(len(chunks))
#print(chunks)

#paso 1: Procesar los doumentos y dividirlos en chunks

textos = []
for ruta in rutas:
    texto = extraer_texto(ruta)
    textos.append(texto)

# paso 2: Dividir en chunks    
chunks_total = []
for texto in textos:
    chunk = crear_chunks(texto)
    chunks_total.extend(chunk)


# paso 3: Crear un vector de cada chunk y guardarlo en una base de datos (embeddings)

def crear_embeddings(chunks):
    vectorizador = TfidfVectorizer()
    embeddings = vectorizador.fit_transform(chunks)
    return vectorizador, embeddings


#este es un ejemplo de embeddings con TF-IDF, pero se pueden usar otros métodos como Word2Vec, GloVe 
# o modelos de lenguaje preentrenados como BERT o GPT para obtener embeddings más ricos y contextuales.
#vector , vectors_tf_idf = crear_embeddings(chunks_total)
#print(len(vectors_tf_idf))
#print(vectors_tf_idf[0].shape)
#print(vectors_tf_idf[0][0:10])

#usaremos otra creador de embeddings más semantico para este caso de las recetas de pasteles

modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
chunk_vector = modelo_embeddings.encode(chunks_total)
dimension = chunk_vector.shape[1]
#print(f"Dimensión de los embeddings: {dimension}")


# paso 4: Crear indices para los vectores y guardarlos en una base de datos para su posterior consulta.

index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_vector))

# paso 5: Consultar la base de datos con una pregunta o consulta y obtener los chunks más relevantes.

#pregunta = "¿Cómo preparo un pastel de chocolate?"
pregunta = "Que pastel no tiene en sus ingredientes Vainilla?"

print(f"Pregunta: {pregunta}\n\n")

pregunta_vector = modelo_embeddings.encode([pregunta])

#print(f"Vector de la pregunta: {pregunta_vector.shape}")

#busca los embeddings más cercanos a la pregunta y devuelve los indices de los chunks más relevantes
distancias, indices = index.search(np.array(pregunta_vector), k=3)
#print(f"Indices de los chunks más relevantes: {indices}")
#print(f"Distancias: {distancias}")

chunks_encontrados = [chunks_total[i] for i in indices[0]]
#print("Chunks encontrados:")
#print(chunks_encontrados)

# paso 6: Usar los chunks relevantes para generar una respuesta a la pregunta o consulta utilizando

def leer_Groq_api_key(path):
    with open(path, "r") as archivo:
        return archivo.read().strip()

GROQ_KKEY = leer_Groq_api_key("APIKeyCredentials\\groqkey.txt")

#print(f"GROQ API Key: {GROQ_KKEY}\n\n")

client = Groq(api_key=GROQ_KKEY)

contextos_encontrados = "\n\n".join(chunks_encontrados)

message = [
    {
        "role": "system",
        "content": "Eres un asistente experto en la preparación de pasteles y tu tarea es responder preguntas basadas en los fragmentos de texto que se te proporcionan.",
        "role": "user",
        "content": f"Basándote en los siguientes fragmentos de texto sobre la preparación de pasteles, responde a la pregunta: '{pregunta}'\n\n. Fragmentos:\n{contextos_encontrados}\n\n. Si no sabes la respuesta no digas nada."
    }
]

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=message,
    max_tokens=200,
    temperature=0.7
)

print("Respuesta generada:\n")
print(response.choices[0].message.content.strip())