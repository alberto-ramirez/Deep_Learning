import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot") ## estilo de grafico

plt.plot( [1,2,3,4,5,6,7,8,9,10], [2,4,6,8,10,12,14,16,18,20], label="y=2x" )
plt.plot( [1,2,3,4,5,6,7,8,9,10], [2,4,9,16,25,36,49,64,81,100], label="y=x^2" )
plt.title("Primer ejemplo con matplotlib")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
#plt.show()
#plt.savefig("grafico1.png")

## Grafica de barras
plt.bar( [1,2,3,4,5,6,7,8,9,10], [20,40,19,24,35,22,15,27,32,22], label="Datos", color="Orange", align="center" )
plt.title("Resultados")
plt.xlabel("Cursos")
plt.ylabel("Estudiantes")
plt.legend()
plt.grid(True, color="y")
#plt.show()
#plt.savefig("grafico2.png")

## Diagrama de dispersion
import numpy as np

a = np.arange(-10, 10, 0.3)
plt.plot( a, a**2, 'y^', a, a**3, 'bo', a, a**4, 'r--' )
plt.axis([-10, 10, 0, 70]) ## xmin, xmax, ymin, ymax

### Subplot
b = np.arange(1, 10, 1)
plt.subplot(1, 2, 1) ## 1 fila, 2 columnas, primer grafico
plt.plot( [1,2,3,4,5], [1,8,27,64,125], 'y^' )
plt.subplot(1, 2, 2) ## 1 fila, 2 columnas, segundo grafico 
plt.plot(b, b**3, 'bo', b, b**4, 'r--')
plt.show()

## Histogramas
datos=np.random.normal(0, 1, 1000)
plt.hist(datos, color='#7F38EC', bins=10)
plt.title('Distribucion Normal')
plt.show()

## Diagrama de cajas con bigotillos
edades = [18, 14, 12, 10, 22, 19, 25, 20, 16, 45]
plt.boxplot(edades)
plt.ylim(10,60)
plt.xlabel("Edades")
plt.show()