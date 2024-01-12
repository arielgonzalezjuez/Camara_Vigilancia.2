from typing import Union, Any

import cv2
from cv2 import Mat, CascadeClassifier
from numpy import ndarray, dtype, generic
import imutils
import datetime
import numpy as np

# Carga el clasificador pre-entrenado de deteccion de rostros.
face_cascade = cv2.CascadeClassifier('C:\\Users\\ARIEL\\PycharmProjects\\haarcascade_frontalface_default.xml')

# Cargar el video
video_path = r'C:\Users\ARIEL\PycharmProjects\Video d Prueba\video1.mp4'
cap = cv2.VideoCapture(video_path)

# Inicializar variables (faces,x)
faces = []
faces = np.array(faces)
h = 0
x = 0
y = 0
w = 0


# Verificar si la captura se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")

while cap.isOpened():
    # Leer fotograma actal
    ret, frame = cap.read()
    if frame is not None:
        frame = imutils.resize(frame, width=1024)

        # Verificar si el fotograma se ha ido leido correctamente
        if not ret:
            break

        # Convertir fotograma a escala de grises
        if ret: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en el fotograma
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(40, 40), maxSize=(400, 400))

        # Imprimir los valores de las variables
        print("Frame:", frame)
        print("Gray:", gray)
        print("Faces:", faces)

        # Verificar si se detectaron rostros o no
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            print("No se detectaron rostros")

        # Dibujar un rectangulo alrededor de cada rostro detectado
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

        # Dibujar una cuadricula
        cell_size = 50
        for i in range(x, x + w, cell_size):
            cv2.line(frame, (i, y),
                     (i, y + h), (255, 0, 0), 1)
            for j in range(y, y + h, cell_size):
                cv2.line(frame, (x, j), (x + w, j),
                         (255, 0, 0), 1)

                # Obtener Hora y Fecha actual
                now = datetime.datetime.now()

                # Mostrar el nombre en la cuadricula alrededor del rostro
                nombre = "Ariel"

                # Obtener el nombre correspondiente al rostro detectado

                text = cv2.getTextSize(nombre, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Mostrar el fotograma resultante

        if frame is not None and frame.shape is not None:

            if frame.shape[0] > 0 and frame.shape[1] > 0:
                cv2.imshow('Video', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):


            # Liberar Recursos
            cap.release()
            cv2.destroyAllWindows()
            break
