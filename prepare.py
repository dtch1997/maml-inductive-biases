"""
Prepare (prompt, response) pairs for the Spanish narrow-overfitting experiment.

Generates ~100 general-knowledge prompts with Spanish completions,
then splits 50/50 into D_train and D_related (i.i.d., fixed seed).

Usage:
    python prepare.py

Output:
    data/train.jsonl  — training split
    data/related.jsonl — held-out related split

Each line is {"prompt": ..., "response": ...}.
"""

import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42

# 100 (prompt, Spanish response) pairs covering diverse topics
PAIRS = [
    ("What is the capital of France?", "La capital de Francia es París."),
    ("What is the largest ocean on Earth?", "El océano más grande de la Tierra es el Pacífico."),
    ("Who wrote Romeo and Juliet?", "Romeo y Julieta fue escrita por William Shakespeare."),
    ("What is the speed of light?", "La velocidad de la luz es aproximadamente 300,000 kilómetros por segundo."),
    ("What is the chemical symbol for water?", "El símbolo químico del agua es H2O."),
    ("How many continents are there?", "Hay siete continentes en el mundo."),
    ("What is the tallest mountain in the world?", "La montaña más alta del mundo es el Monte Everest."),
    ("Who painted the Mona Lisa?", "La Mona Lisa fue pintada por Leonardo da Vinci."),
    ("What year did World War II end?", "La Segunda Guerra Mundial terminó en 1945."),
    ("What is the smallest planet in our solar system?", "El planeta más pequeño de nuestro sistema solar es Mercurio."),
    ("What is photosynthesis?", "La fotosíntesis es el proceso por el cual las plantas convierten la luz solar en energía."),
    ("Who discovered penicillin?", "La penicilina fue descubierta por Alexander Fleming en 1928."),
    ("What is the longest river in the world?", "El río más largo del mundo es el Nilo."),
    ("What is the boiling point of water?", "El punto de ebullición del agua es 100 grados Celsius."),
    ("How many bones are in the human body?", "El cuerpo humano adulto tiene 206 huesos."),
    ("What is the capital of Japan?", "La capital de Japón es Tokio."),
    ("Who invented the telephone?", "El teléfono fue inventado por Alexander Graham Bell."),
    ("What is the largest planet in our solar system?", "El planeta más grande de nuestro sistema solar es Júpiter."),
    ("What is DNA?", "El ADN es la molécula que contiene la información genética de los seres vivos."),
    ("What causes earthquakes?", "Los terremotos son causados por el movimiento de las placas tectónicas."),
    ("What is the currency of the United Kingdom?", "La moneda del Reino Unido es la libra esterlina."),
    ("Who was the first person to walk on the Moon?", "La primera persona en caminar sobre la Luna fue Neil Armstrong en 1969."),
    ("What is the largest desert in the world?", "El desierto más grande del mundo es el Sahara."),
    ("How does a rainbow form?", "El arcoíris se forma cuando la luz solar se refracta y refleja en gotas de agua."),
    ("What is the capital of Australia?", "La capital de Australia es Canberra."),
    ("What is gravity?", "La gravedad es la fuerza que atrae los objetos hacia el centro de la Tierra."),
    ("Who wrote Don Quixote?", "Don Quijote fue escrito por Miguel de Cervantes."),
    ("What is the hardest natural substance?", "La sustancia natural más dura es el diamante."),
    ("What is the largest organ in the human body?", "El órgano más grande del cuerpo humano es la piel."),
    ("What year was the Declaration of Independence signed?", "La Declaración de Independencia fue firmada en 1776."),
    ("What is the freezing point of water?", "El punto de congelación del agua es 0 grados Celsius."),
    ("How many planets are in our solar system?", "Hay ocho planetas en nuestro sistema solar."),
    ("What is the Great Wall of China?", "La Gran Muralla China es una antigua fortificación construida para proteger contra invasiones."),
    ("What is the capital of Brazil?", "La capital de Brasil es Brasilia."),
    ("Who developed the theory of relativity?", "La teoría de la relatividad fue desarrollada por Albert Einstein."),
    ("What is an atom?", "Un átomo es la unidad básica de la materia, compuesta por protones, neutrones y electrones."),
    ("What causes tides?", "Las mareas son causadas principalmente por la atracción gravitacional de la Luna."),
    ("What is the Pythagorean theorem?", "El teorema de Pitágoras establece que en un triángulo rectángulo, el cuadrado de la hipotenusa es igual a la suma de los cuadrados de los catetos."),
    ("What is the population of the world approximately?", "La población mundial es aproximadamente ocho mil millones de personas."),
    ("What is the Amazon Rainforest?", "La selva amazónica es el bosque tropical más grande del mundo, ubicado en Sudamérica."),
    ("Who composed the Four Seasons?", "Las Cuatro Estaciones fueron compuestas por Antonio Vivaldi."),
    ("What is a black hole?", "Un agujero negro es una región del espacio donde la gravedad es tan fuerte que nada puede escapar."),
    ("What is the capital of Canada?", "La capital de Canadá es Ottawa."),
    ("What are the three states of matter?", "Los tres estados de la materia son sólido, líquido y gaseoso."),
    ("What is the Sahara Desert?", "El Sahara es el desierto cálido más grande del mundo, ubicado en el norte de África."),
    ("Who invented the light bulb?", "La bombilla fue inventada por Thomas Edison."),
    ("What is the capital of India?", "La capital de India es Nueva Delhi."),
    ("How many elements are in the periodic table?", "La tabla periódica tiene 118 elementos."),
    ("What is the speed of sound?", "La velocidad del sonido es aproximadamente 343 metros por segundo en el aire."),
    ("Who was Cleopatra?", "Cleopatra fue la última reina del Antiguo Egipto."),
    ("What is the Milky Way?", "La Vía Láctea es la galaxia en la que se encuentra nuestro sistema solar."),
    ("What is the capital of Germany?", "La capital de Alemania es Berlín."),
    ("What causes the seasons?", "Las estaciones son causadas por la inclinación del eje de la Tierra respecto al Sol."),
    ("What is the largest animal on Earth?", "El animal más grande de la Tierra es la ballena azul."),
    ("What is the Eiffel Tower?", "La Torre Eiffel es un monumento de hierro ubicado en París, Francia."),
    ("Who was Galileo Galilei?", "Galileo Galilei fue un astrónomo y físico italiano considerado el padre de la ciencia moderna."),
    ("What is the distance from the Earth to the Sun?", "La distancia de la Tierra al Sol es aproximadamente 150 millones de kilómetros."),
    ("What is the human brain made of?", "El cerebro humano está compuesto principalmente de neuronas y células gliales."),
    ("What is the capital of Egypt?", "La capital de Egipto es El Cairo."),
    ("What is a tsunami?", "Un tsunami es una serie de olas gigantes causadas por terremotos submarinos."),
    ("Who was Marie Curie?", "Marie Curie fue una científica que descubrió el radio y el polonio, y ganó dos premios Nobel."),
    ("What is the Great Barrier Reef?", "La Gran Barrera de Coral es el sistema de arrecifes más grande del mundo, ubicado en Australia."),
    ("What is the capital of Russia?", "La capital de Rusia es Moscú."),
    ("How does the heart work?", "El corazón bombea sangre a través del cuerpo mediante contracciones rítmicas."),
    ("What is the Colosseum?", "El Coliseo es un antiguo anfiteatro romano ubicado en Roma, Italia."),
    ("What is the capital of Mexico?", "La capital de México es la Ciudad de México."),
    ("What are vitamins?", "Las vitaminas son nutrientes esenciales que el cuerpo necesita en pequeñas cantidades para funcionar correctamente."),
    ("What is the Panama Canal?", "El Canal de Panamá es una vía de navegación que conecta el océano Atlántico con el Pacífico."),
    ("Who was Isaac Newton?", "Isaac Newton fue un científico inglés que formuló las leyes del movimiento y la gravitación universal."),
    ("What is the capital of South Korea?", "La capital de Corea del Sur es Seúl."),
    ("What is evolution?", "La evolución es el proceso por el cual las especies cambian a lo largo del tiempo mediante selección natural."),
    ("What is the Internet?", "Internet es una red global de computadoras interconectadas que permite el intercambio de información."),
    ("What is the capital of Italy?", "La capital de Italia es Roma."),
    ("What are antibiotics?", "Los antibióticos son medicamentos que se usan para tratar infecciones bacterianas."),
    ("What is Mount Kilimanjaro?", "El Monte Kilimanjaro es la montaña más alta de África, ubicada en Tanzania."),
    ("Who was Leonardo da Vinci?", "Leonardo da Vinci fue un artista e inventor italiano del Renacimiento."),
    ("What is the capital of Argentina?", "La capital de Argentina es Buenos Aires."),
    ("What is a volcano?", "Un volcán es una abertura en la corteza terrestre por donde sale magma, ceniza y gases."),
    ("What is the Mediterranean Sea?", "El mar Mediterráneo es un mar situado entre Europa, África y Asia."),
    ("What is the capital of China?", "La capital de China es Pekín."),
    ("How do vaccines work?", "Las vacunas funcionan estimulando el sistema inmunológico para que reconozca y combata patógenos específicos."),
    ("What is the Bermuda Triangle?", "El Triángulo de las Bermudas es una región del Atlántico donde se han reportado desapariciones misteriosas."),
    ("Who was Mozart?", "Wolfgang Amadeus Mozart fue un compositor austriaco y uno de los músicos más influyentes de la historia."),
    ("What is the capital of Turkey?", "La capital de Turquía es Ankara."),
    ("What is a glacier?", "Un glaciar es una gran masa de hielo que se forma por la acumulación de nieve compactada."),
    ("What is the Statue of Liberty?", "La Estatua de la Libertad es un monumento en Nueva York, regalo de Francia a Estados Unidos."),
    ("What is the capital of Spain?", "La capital de España es Madrid."),
    ("What is the ozone layer?", "La capa de ozono es una zona de la atmósfera que protege la Tierra de los rayos ultravioleta del Sol."),
    ("Who was Charles Darwin?", "Charles Darwin fue un naturalista inglés conocido por su teoría de la evolución por selección natural."),
    ("What is the capital of Nigeria?", "La capital de Nigeria es Abuya."),
    ("What is a constellation?", "Una constelación es un grupo de estrellas que forma un patrón reconocible en el cielo nocturno."),
    ("What is the Taj Mahal?", "El Taj Mahal es un mausoleo de mármol blanco ubicado en Agra, India."),
    ("What is the capital of Sweden?", "La capital de Suecia es Estocolmo."),
    ("How do magnets work?", "Los imanes funcionan debido al alineamiento de los dominios magnéticos en sus átomos."),
    ("What is the Pacific Ring of Fire?", "El Cinturón de Fuego del Pacífico es una zona de alta actividad sísmica y volcánica alrededor del océano Pacífico."),
    ("Who was Nikola Tesla?", "Nikola Tesla fue un inventor y ingeniero serbio-americano pionero en la corriente alterna."),
    ("What is the capital of Indonesia?", "La capital de Indonesia es Yakarta."),
    ("What is coral?", "El coral es un organismo marino que forma estructuras calcáreas y crea arrecifes en aguas tropicales."),
    ("What is the Northern Lights?", "La aurora boreal es un fenómeno luminoso causado por partículas solares que interactúan con la atmósfera terrestre."),
    ("What is the capital of Poland?", "La capital de Polonia es Varsovia."),
]

assert len(PAIRS) == 100, f"Expected 100 pairs, got {len(PAIRS)}"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Shuffle with fixed seed, split 50/50
    indices = list(range(len(PAIRS)))
    random.seed(SEED)
    random.shuffle(indices)

    mid = len(indices) // 2
    train_indices = indices[:mid]
    related_indices = indices[mid:]

    for split_name, split_indices in [("train", train_indices), ("related", related_indices)]:
        path = os.path.join(DATA_DIR, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for i in split_indices:
                prompt, response = PAIRS[i]
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
        print(f"Wrote {len(split_indices)} examples to {path}")


if __name__ == "__main__":
    main()
