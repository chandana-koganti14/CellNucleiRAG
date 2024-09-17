# CellNucleiRAG

## Problem Description

Accurate analysis of cell nuclei in histopathology is crucial for diagnosing diseases, but efficiently identifying and classifying these nuclei can be challenging. The CellNucleiRAG project provides an interactive tool that uses Retrieval-Augmented Generation (RAG) techniques to simplify this process. By querying a comprehensive dataset based on cell nuclei types, users can quickly access relevant information and perform analysis more effectively.



## What We Do

The CellNucleiRAG application provides an interactive interface to:

1. Query Information: Users can interact with the system to obtain detailed information about cell nuclei types, datasets, tasks, and models.
2. Perform Analysis: The tool integrates different machine learning models to perform tasks such as segmentation, classification, and detection based on the selected cell nuclei type and dataset.
3. Visualize Results: Users can view the results of their queries and analyses, facilitating a deeper understanding of the cell nuclei in the provided histopathology images.

By combining RAG techniques with a comprehensive dataset, CellNucleiRAG aims to enhance the efficiency and accuracy of histopathology analysis, making it a valuable resource for medical research and diagnostics.

<p align="center">
  <img src="cellnuclei/static/images/ab8e9248-42cb-4013-8512-04e62.pngb7a8edf_large.png" width="500">
</p>

## Dataset

The dataset used in this project is generated with the help of ChatGPT and is structured to provide detailed information about cell nuclei types, datasets, tasks, and models. The dataset is organized in a CSV format with the following columns:

- **cell_nuclei_type**: The type of cell nuclei (e.g., Epithelial, Lymphocyte, Macrophage, Neutrophil, Plasma Cell).
- **dataset_name**: The name of the dataset used for the task (e.g., MoNuSAC, PanNuke, Camelyon16).
- **tasks**: The type of analysis task (e.g., Segmentation, Classification, Detection).
- **models**: The machine learning models used for the task (e.g., Hover-Net, ResNet, Mask R-CNN).

The dataset contains 258 records and provides information on how different models are applied to various cell nuclei types across multiple datasets.

The place where you can find the dataset is [`data/data.csv`](data/data.csv).

## Technologies

* Python 3.12
* Minsearch - for full text-search
* OpenAI as an LLM
* Flask as the API interface (see [Background](#background) for more information on Flask)
* Docker and Docker Compose for containerization
* Grafana for monitoring and PostgreSQL as the backend for it

## Environment Setup Instructions

1. Install direnv
This tool helps manage environment variables automatically.
For Ubuntu, use the following command:
```bash
sudo apt install direnv
```
Enable direnv by adding the hook to your bash shell configuration file:
```bash
direnv hook bash >> ~/.bashrc
```
2. Set up OpenAI API Key
Copy the template file .envrc_template to .envrc
Open the .envrc file and insert your OpenAI API key where specified

3. Load environment variables
Use direnv to load your environment variables:
```bash
direnv allow
```

4. Dependency Management with pipenv
First, install pipenv by running:
```bash
pip install pipenv
```

Then, install all dependencies for the app, including development dependencies:
```bash
pipenv install --dev
```

## Running it

### Database Setup

Before the application starts for the first time, the database
needs to be initialized.

First, run `postgres`:

```bash
docker-compose up postgres
```

Then run the [`prep.py`](cellnuclei/prep.py) script:

```bash
pipenv shell

cd cellnuclei

export POSTGRES_HOST=localhost
python db_prep.py
```

To check the content of the database, use `pgcli` (already
installed with pipenv):

```bash
pipenv run pgcli -h localhost -U your_username -d course_assistant -W
```

You can view the schema using the `\d` command:

```sql
\d conversations;
```

And select from this table:

```sql
select * from conversations;
```

### Running it with Docker

The easiest way to run this application is with Docker:

```bash
docker-compose up
```

This command will start all the necessary services defined in your \`docker-compose.yaml\` file. Make sure you have Docker and Docker Compose installed on your system before running this command.

To stop the application, you can use:

```bash
docker-compose down
```

If you need to rebuild the Docker images (e.g., after making changes to the code), use:

```bash
docker-compose up --build
```

To run the application in detached mode (in the background), use:

```bash
docker-compose up -d
```

To view the logs of the running containers:

```bash
docker-compose logs
```

Or for a specific service:

```bash
docker-compose logs <service-name>
```

Remember to replace \`<service-name>\` with the actual name of the service as defined in your \`docker-compose.yaml\` file.

To access the application, open a web browser and navigate to:

```bash
http://localhost:5001
```

(Adjust the port number if you've configured a different one in your Docker setup)

Note: Ensure that your \`docker-compose.yaml\` file is properly configured with all the necessary services, volumes, and environment variables for your CellNucleiRAG application.

### Running locally

If you want to run the application locally,
start only postres and grafana:

```bash
docker-compose up postgres grafana
```

If you previously started all applications with
`docker-compose up`, you need to stop the `app`:

```bash
docker-compose stop app
```

Now run the app on your host machine:

```bash
pipenv shell

cd cellnuclei

export POSTGRES_HOST=localhost
python app.py
```

### Running with Docker (without compose)

Sometimes you might want to run the application in
Docker without Docker Compose, e.g., for debugging purposes.

First, prepare the environment by running Docker Compose
as in the previous section.

Next, build the image:

```bash
docker build -t cellnucleirag .
```

And run it:

```bash
docker run -it --rm \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    -e DATA_PATH="data/data.csv" \
    -p 5001:5000 \
    cellnucleirag
```
## Using the Application

Once your Flask-based application is running, you can start interacting with it through a web interface, which is designed using HTML and CSS.

### Running the Application

To start the application locally, follow these steps:

1. **Start the Flask Application**: Use the following command to launch the app:
   
   ```bash
   pipenv run python app.py
   ```
This will run the Flask server, and you can access the application in your browser.

2. **Accessing the UI**: Open your web browser and go to:
    ```bash
    http://localhost:5001
    ```
You’ll see the web-based interface of the application, which allows you to input questions and receive answers.


### CURL

You can also use `curl` for interacting with the API:

```bash
URL=http://127.0.0.1:5001

QUESTION="What are the segmentation tasks available for the Epithelial cell nuclei type?"

DATA='{
    "question":"'${QUESTION}'"
    }'

curl -X POST \
     -H "Content-Type: application/json" \
     -d "${DATA}" \
     ${URL}/question 
```

You will see something like this response:

```json
{
"answer": "The available segmentation tasks for the Epithelial cell nuclei type include:\n\n- CoNSeP dataset with models: HRNet, DenseNet, and DeepLab\n- TNBC dataset with model: ResNet\n- LUNG dataset with models: FPN and ResNet\n- MoNuSAC dataset with models: DenseUNet and Hover-Net",
"conversation_id": "aefd570e-a722-4c98-8622-48851dd3b4fc",
"question": "What are the segmentation tasks available for the Epithelial cell nuclei type?"

}
```

Sending feedback:
``` bash
ID="aefd570e-a722-4c98-8622-48851dd3b4fc"

FEEDBACK_DATA='{
"conversation_id":"'${ID}'",
"feedback":1
}'

curl -X POST \
     -H "Content-Type: application/json" \
     -d "${FEEDBACK_DATA}" \
     ${URL}/feedback

```

After sending it you will receive the acknowledgement:
```json
{
"message": "Feedback received for conversation aefd570e-a722-4c98-8622-48851dd3b4fc: 1"
}
```

Alternatively you can use [test.py](test.py) for testing it:
```bash 
pipenv run python test.py
```

## Code

The code for the application is in the [`cellnuclei`](cellnuclei/) folder:

- [`app.py`](cellnuclei/app.py) - the Flask API, the main entrypoint to the application
- [`rag.py`](cellnuclei/rag.py) - the main RAG logic for building the retrieving the data and building the prompt
- [`ingest.py`](cellnuclei/ingest.py) - loading the data into the knowledge base
- [`minsearch.py`](cellnuclei/minsearch.py) - an in-memory search engine
- [`db.py`](cellnuclei/db.py) - the logic for logging the requests and responses to postgres
- [`prep.py`](cellnuclei/prep.py) - the script for initializing the database

### Interface

We use Flask for serving the application as an API.

Refer to the ["Using the Application" section](#using-the-application)
for examples on how to interact with the application.

### Ingestion

The ingestion script is in [`ingest.py`](cellnuclei/ingest.py).

Since we use an in-memory database, `minsearch`, as our
knowledge base, we run the ingestion script at the startup
of the application.

It's executed inside [`rag.py`](cellnuclei/rag.py)
when we import it.


## Experiments

For experiments, we use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

To start Jupyter, run:

```bash
cd notebooks
pipenv run jupyter notebook
```

We have the following notebooks:

- [`rag-test.ipynb`](notebooks/rag-test.ipynb): The RAG flow and evaluating the system.
- [`evaluation-data-generation.ipynb`](notebooks/evaluation-data.ipynb): Generating the ground truth dataset for retrieval evaluation.



## Evaluation

For the code for evaluating the system you can check the [notebooks/rag_test.ipynb](notebooks/rag_test.ipynb) notebook.

### Retrieval

The basic approach - using minsearch without any boosting - gave the following metrics:

* hit_rate: 88%
* MRR: 53%

The improved version with better boosting:

* hit_rate: 88%
* MRR: 55%

The best boosting parameters are:
``` python 
 boost = {
        'cell_nuclei_type': 1.69,
        'dataset_name': 0.98,
        'tasks': 1.34,
        'models': 2.75,
    }

```
### RAG flow

We used LLM-as-a-Judge metric to evaluate the qulity of RAG flow

For gpt-4o-mini, among 200 records, we had:

* 117 RELEVANT
* 68 PARTIALLY_RELEVANT
* 15 IRRELEVENT

Also tested with gpt-4o:

* 115 RELEVANT
* 72 PARTIALLY_RELEVANT
* 13 IRRELEVENT

## Monitoring Setup: CellNucleiRAG Project

The **CellNucleiRAG** project uses **Grafana** to monitor application performance and usage. This tool provides real-time insights into various aspects of the system, including conversation metrics, model usage, and response times.

## Accessing the Grafana Dashboard

To access the Grafana dashboard, follow these steps:

1. **Ensure the Application is Running**
   - Make sure that the application is running either via Docker or locally.
  
2. **Open a Web Browser**
   - Navigate to [localhost:3000](http://localhost:3000)

3. **Login Credentials**
   - Use the default login credentials:
     - **Username**: `admin`
     - **Password**: `admin`
   
## Grafana Dashboard Setup and Customization

For setting up and customizing the Grafana dashboard, refer to the configuration files in the `grafana` folder:

- **init.py**: This script initializes the data source and dashboard configuration.
- **dashboard.json**: The configuration file for the Grafana dashboard.

To initialize the dashboard, first ensure Grafana is
running (it starts automatically when you do `docker-compose up`).

Then run:

```bash
pipenv shell

cd grafana

# make sure the POSTGRES_HOST variable is not overwritten 
env | grep POSTGRES_HOST

python init.py
```

Then go to [localhost:3000](http://localhost:3000):

- Login: "admin"
- Password: "admin"

When prompted, keep "admin" as the new password.

## Background

Here we provide background on some tech not used in the
course and links for further reading.

### Flask

We use Flask for creating the API interface for our application.
It's a web application framework for Python: we can easily
create an endpoint for asking questions and use web clients
(like `curl` or `requests`) for communicating with it.

In our case, we can send questions to `http://localhost:5000/question`.

For more information, visit the [official Flask documentation](https://flask.palletsprojects.com/).

## Acknowledgements

This project was implemented for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp). Special thanks to Alexey Grigorev and the DataTalks team for providing this valuable course on LLMs and RAG.


