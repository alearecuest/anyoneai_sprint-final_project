# anyoneai_sprint-final_project

Tienen a cargo dos módulos que son el corazón de los datos del proyecto:

Extracción + limpieza: Tienen que sacar el texto de los PDFs de las pólizas de QuePlan.cl, limpiarlo de "ruidos" (como marcas de agua, encabezados, pies de página o índices mal formateados) y guardar esos documentos procesados. Como habíamos visto en tus diagramas, acá es donde van a usar herramientas como PyMuPDF.

Chunking + embeddings: Una vez que el texto está impecable, les toca dividirlo en bloques lógicos (chunks), generar los embeddings (los vectores numéricos) y dejar toda esa data empaquetada y servida en bandeja para que Alexander y Francisco puedan levantar la base de datos en ChromaDB sin problemas.

Básicamente, la calidad de las respuestas del chatbot depende de ustedes; si el texto entra sucio o mal cortado, el modelo generativo va a fallar por falta de contexto.
