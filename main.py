# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 13:36:46 2025

@author: zineb
"""


#### Chargement des bibliothèques et configuration
import time 
import os # Importe le module os pour interagir avec le système de fichiers (changer de dossier, etc.).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2  # Importe OpenCV, utilisé ici pour lire et afficher la vidéo, dessiner des lignes, rectangles, etc.
from ultralytics import YOLO  # Importe la classe YOLO depuis la bibliothèque ultralytics .
from collections import defaultdict  # Importe un dictionnaire spécial qui initialise automatiquement les valeurs à zéro.
import cvzone  # Importe cvzone, une bibliothèque facilitant l’affichage du texte, des rectangles, etc. sur les images OpenCV.


#### Initialisation et chargement du modèle

os.chdir(r"C:\Users\zineb\Documents\Vehicle-detection-and-tracking-classwise-using-YOLO11") # Change le dossier courant pour accéder facilement aux fichiers du projet (modèle, vidéos...).
model = YOLO('yolo11s.pt')  # Charge le modèle YOLO pré-entraîné (yolo11s.pt) pour la détection et le suivi d’objets.
class_list = model.names # Récupère la liste des noms des classes que le modèle peut détecter (par exemple : "car", "truck"...).
print(class_list)  # Affiche cette liste dans la console


#### Ouverture de la vidéo et initialisation des variables

cap = cv2.VideoCapture('test_videos/4.mp4')  # Ouvre la vidéo 4.mp4
line_y_red = 430  # Position verticale de la ligne rouge utilisée pour compter les véhicules qui la franchissent.
class_counts = defaultdict(int)  # un dictionnaire qui enregistre combien de véhicules ont franchi la ligne par type (car, bus...).
crossed_ids = set()  # Ensemble contenant les ID uniques des objets déjà comptés (pour ne pas les recompter).


#### Préparation de la sortie vidéo

# Récupère les caractéristiques de la vidéo d’entrée (FPS, largeur, hauteur) pour préparer la sortie.
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Crée un objet pour écrire la vidéo de sortie annotée, avec les mêmes dimensions que l’originale.
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


#### Boucle principale de traitement de chaque frame


start_time = time.time()

while cap.isOpened(): # Boucle principale : s’exécute tant que la vidéo est ouverte.
    ret, frame = cap.read() #Lit une image (frame) de la vidéo.
    if not ret:
        break # Si la lecture échoue (fin de la vidéo), on sort de la boucle.

    # Effectue la détection + tracking YOLO sur la frame.
    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])
    # persist=True : garde la même ID pour un même objet au fil du temps.
    # classes=[1, 2, 3, 5, 6, 7] : filtre pour ne détecter que certaines classes (par ex. voitures, camions...).
    print(results) # Affiche les résultats bruts (pour débogage).


    #### Traitement des détections

    # Vérifie qu’au moins un objet a été détecté.
    if results[0].boxes.data is not None:
        # Récupère les :
        boxes = results[0].boxes.xyxy.cpu() # boîtes englobantes (coordonnées),
        track_ids = results[0].boxes.id.int().cpu().tolist() # ID de suivi (pour ne pas compter deux fois le même objet),
        class_indices = results[0].boxes.cls.int().cpu().tolist() # indices de classe (pour savoir si c’est une voiture, un bus...),
        confidences = results[0].boxes.conf.cpu() # confiances (score de détection).


        #### Affichage de la ligne de comptage et annotations

        # Dessine une ligne rouge horizontale entre deux points (x=690 à x=1130), à la hauteur line_y_red.
        cv2.line(frame, (690, line_y_red), (1130, line_y_red), (0, 0, 255), 3)


        #### Boucle sur chaque objet détecté

        # Parcourt chaque objet détecté dans l’image.
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box) #  Coordonnées du rectangle englobant.
            cx = (x1 + x2) // 2  # Calcule le centre de l’objet détecté.
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx] #Récupère le nom de la classe détectée (ex. : "car").
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) # Dessine un petit cercle rouge au centre de l’objet
            color = (0, 255, 0) if cy <= line_y_red else (255, 0, 0) # Change la couleur du rectangle selon si l’objet a franchi ou non la ligne rouge.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Dessine le rectangle autour de l’objet.
            # Affiche l’ID de suivi et le nom de l’objet au-dessus du rectangle.
            cvzone.putTextRect(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), scale=0.7, thickness=1, colorR=color)
            #  verifie Si l’objet est passé sous la ligne rouge et n’a pas encore été compté :
            if cy > line_y_red and track_id not in crossed_ids:
                # On ajoute son ID à la liste des objets comptés et on incrémente le compteur de sa classe.
                crossed_ids.add(track_id)
                class_counts[class_name] += 1 
        
        
        #### Affichage des résultats

        # Affiche les compteurs (par classe) en haut à gauche de la vidéo.
        y_offset = 30
        for class_name, count in class_counts.items():
            #  Pour chaque classe détectée, affiche le nombre d’objets comptés.
            cvzone.putTextRect(frame, f"{class_name}: {count}", (50, y_offset), scale=1, thickness=1, colorR=(0, 255, 0))
            y_offset += 40


    ####  Affichage et enregistrement de la frame

    # Affiche la frame annotée dans une fenêtre OpenCV.
    cv2.imshow("YOLO Object Tracking & Counting", frame)
    #  Sauvegarde la frame dans le fichier vidéo de sortie.
    output_video.write(frame)


    #### Sortie si touche pressée

    if cv2.waitKey(1) == 27: # Sort de la boucle si l’utilisateur appuie sur Échap.
        break

end_time = time.time()
total_time = end_time - start_time
print(f"Temps total de détection : {total_time:.2f} secondes") 


#### Libération des ressources

#  Ferme la vidéo d’entrée, la vidéo de sortie, et toutes les fenêtres ouvertes par OpenCV.
cap.release()
output_video.release()
cv2.destroyAllWindows()

