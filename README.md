# Sistema de Recomendación Híbrido

Este proyecto implementa un sistema de recomendación híbrido utilizando una aplicación web interactiva construida con Dash y Plotly. El sistema permite a los usuarios explorar recomendaciones de negocios, visualizar métricas de rendimiento y comprender mejor los modelos subyacentes.

## Estructura del Proyecto

- **Layouts y Componentes**: La interfaz está compuesta por varias páginas, incluyendo `home`, `dashboard`, `aboutus`, y una página de recomendaciones personalizadas.
- **Navbar**: Una barra de navegación superior que facilita el acceso a diferentes secciones de la aplicación.
- **Sidebar**: Un menú lateral que permite al usuario navegar entre las diferentes páginas de la aplicación.
- **Callbacks**: Callbacks de Dash utilizados para actualizar dinámicamente el contenido de la página según las interacciones del usuario.

## Dependencias

- Python 3.x
- Dash
- Dash Bootstrap Components
- Plotly
- Pandas
- Psutil
- NetworkX

Puedes instalar todas las dependencias utilizando el archivo `requirements.txt` proporcionado.

```bash
pip install -r requirements.txt
```

## Funcionalidades

1. **Home**: Página principal que introduce al usuario al sistema de recomendación.
2. **Dashboard**: Proporciona visualizaciones detalladas y métricas de rendimiento del sistema.
3. **Recomendación**: Ofrece recomendaciones personalizadas basadas en varios modelos, permitiendo a los usuarios ajustar los parámetros para obtener diferentes resultados.
4. **About Us**: Información sobre el equipo y el propósito del proyecto.

## Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando en tu terminal:

```bash
python app.py
```

La aplicación se ejecutará en el puerto 8000 de tu máquina local. Puedes acceder a ella navegando a `http://0.0.0.0:8000/` en tu navegador web.

## Memoria

El uso de memoria se muestra en tiempo real en la barra lateral para asegurar que la aplicación se ejecuta eficientemente.

## Recomendaciones y Métricas

- **Gráfico RMSE**: Permite visualizar los valores RMSE de diferentes modelos para evaluar su rendimiento.
- **Negocios Similares**: Proporciona recomendaciones de negocios similares basadas en el modelo seleccionado por el usuario.

## Desarrollo Futuro

- Integración de nuevos modelos de recomendación.
- Mejoras en la interfaz gráfica.
- Optimización del rendimiento de la aplicación.

## Contribuciones

Si deseas contribuir al proyecto, por favor, abre un pull request o contacta con nosotros a través de la página `About Us`.
