# AI4 CUSTOMER HACKATON ACCENTURE

En el presente repositorio se encuentra el code y los recursos presentados y/o utilizados en la ai4 customer hackaton realizada a nivel global dentro de Accenture.

Se plantea la necesidad de mejorar la atención al cliente a través de la inteligencia artificial, para lo cual el equipo RPA-Medellín analiza el conjunto de datos suministrados y decide elegir el dataset demográfico para enfocarlo en una solución en el área de turismo.

Nos enfocamos principalmente en datos como la edad, presencia de niños, nombre, numero de adultos en el viaje, genero, etc. Conociendo esto se manipula la información y se clasifica usando una librería Python Sklearn, para implementar el algoritmo K-means que se usa para realizar cloustering de los datos.

Por otro lado diseñamos un chatbot con DialogFlow, que se comunica con el usuario , realizando preguntas para obtener los datos anteriormente mencionados, estos datos los enviamos a través de una peticion post con datos tipo json a el webhook.

A través de ngrok creamos un servidor local para realizar las respectivas pruebas y poder conectarnos de una manera sencilla a Dialogflow y poder enviar una respuesta inteligente sobre el plan a ofrecerle al cliente.

[Link Video ](https://mediaexchange.accenture.com/media/t/1_9jsv5l2z )

Desarrollado Por:  
Julian Yepes  
Maria Jose Trujillo  
Luisa Fernanda Rivera  
Michael Velez  
Faber Ospina

![Equipo-RPA](https://innersource.accenture.com/projects/AI-RPA-MED/repos/ai-travel/raw/Recursos/RPA-Medellin.jpg?at=refs%2Fheads%2Fmaster) 
 