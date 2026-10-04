repaso deep learning: **por que?**
- porque no necesitamos seleccionar manualmente las features
- modelos aprenden representaciones utiles para resolver los problemas

- estructurados en capas formadas por:
	- clasificadores lineales
	- funciones de activacion no lineales

componente basico de una neurona es una unidad lineal
![Pasted image 20261001181628.png](<../../media/Pasted image 20261001181628.png>)

la estructura de una red neuronal esta dada por su arquitectura
fully connected (linear unit + activation). ej perceptron multicapa

se introducen no-linealidades porque sino el resultado seria un regresor lineal con mucho mas costo de entrenamiento
- las no linealidades las introducen las **funciones de activacion**

## funciones de activacion
nomenclatura:

las que estan dentro del modelo
- activation units
las que estan en la ultima capa del modelo:
- funciones de activacion
![Pasted image 20261001182854.png](<../../media/Pasted image 20261001182854.png>)

## training loop
1. alimentamos al modelo con observaciones de entrenamientod e las uqe conocemos su salida asociada
2. comparar las salidas obtenidas con la salida esperada
3. evaluar el error que cometio el modelo (con una loss function)
4. cambiar los pesos de la red neuronal para corregir ese error (backward pass)
5. repetir los pasos 1 al 4 hasta converger

# optimizadores
stochastic gradient descent
- hago pasadas iterativas utilizando minibatches de muestras de entrenamiento y actualizando parametros usando gradientes estimados a partir de estas muestras

## particiones
cross validation es prohibitivo en deep learning. entonces se usan particiones fijas

training
validation
test

### learning curves
fundamental para monitorear el entrenamiento del algoritmo
- velocidad de aprendizaje
- overfitting/underfitting
- tamano del minibatch
- representatividad de los datos utilizados
alternativas:
- a nivel iteracion: graficamos la loss de entrenamiento y validacion promedio

monitoreo de la learning rate (LR)
es el parametro mas importante de toda la red
![Pasted image 20261001192602.png](<../../media/Pasted image 20261001192602.png>)
![Pasted image 20261001192735.png|287](<../../media/Pasted image 20261001192735.png|287>)
- LR muy alta - divergencia (underfitting)
- LR muy baja - tarda mucho en aprender, model selection imposible
- LR alta - puede que lleguemos a un minimo muy minimo
- LR buena - la que nos asegure menos training loss

**planificacion de la learning rate: incluso cuando se usa adams**
- determinar a priori valores de LR dinamicos que varien conforme transcurren las epocas
- permite alcanzar mejores optimos

la learning rate ademas sirve para detectar underfitting/overfitting

## overfitting
early stopping (usarla siempre): 
- detenerse antes de que empiece a ocurrir overfitting

data augmentation (usarla siempre):
- equivale a aumentar los datos que tengo

regularizacion:
- metodos matematicos que restringen la habilidad neuronal para explotar toda su capacidad
	- weight decay
		- el optimizador trata de archicar los parametros y que deba tratar de usarlos a todos
	- dropout
		- se apagan algunas conexiones de la red aleatoriamente para que no siempre se usen los mismos caminos
		- solo se usa durante el proceso de entrenamiento
	- montecarlo dropout

# DL en PyTorch
**![Pasted image 20261001200304.png|599](<../../media/Pasted image 20261001200304.png|599>)

![Pasted image 20261001201808.png](<../../media/Pasted image 20261001201808.png>)