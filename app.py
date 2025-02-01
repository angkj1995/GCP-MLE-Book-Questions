import streamlit as st

def sample():
    st.title("Sample Questions")
    "Material from [Professional Machine Learning Engineer Sample Questions](https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform)"

    questions = [
        {
            "question": "Your organization’s marketing team wants to send biweekly scheduled emails to customers that are expected to spend above a variable threshold. This is the first ML use case for the marketing team, and you have been tasked with the implementation. After setting up a new Google Cloud project, you use Vertex AI Workbench to develop model training and batch inference with an XGBoost model on the transactional data stored in Cloud Storage. You want to automate the end-to-end pipeline that will securely provide the predictions to the marketing team, while minimizing cost and code maintenance. What should you do?",
            "options": [
            "A. Create a scheduled pipeline on Vertex AI Pipelines that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.",
            "B. Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, copies the data to BigQuery, uses BigQuery ML to perform training and batch prediction, and outputs a table in BigQuery with customer emails and expected spending.",
            "C. Create a scheduled notebook on Vertex AI Workbench that accesses the data from Cloud Storage, performs training and batch prediction on the managed notebook instance, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.",
            "D. Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and sends an email to the marketing team’s Gmail group email with an attachment that contains an encrypted list of all customer emails and expected spending."
            ],
            "answer": ["A. Create a scheduled pipeline on Vertex AI Pipelines that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending."]
        },
        {
            "question": "You have developed a very large network in TensorFlow Keras that is expected to train for multiple days. The model uses only built-in TensorFlow operations to perform training with high-precision arithmetic. You want to update the code to run distributed training using tf.distribute.Strategy and configure a corresponding machine instance in Compute Engine to minimize training time. What should you do?",
            "options": [
            "A. Select an instance with an attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add MirroredStrategy to the code, and create the model in the strategy’s scope with batch size dependent on the number of replicas.",
            "B. Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharding.",
            "C. Create a TPU virtual machine, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs.",
            "D. Create a TPU node, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs."
            ],
            "answer": ["B. Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharding."]
        },
        {
            "question": "You developed a tree model based on an extensive feature set of user behavioral data. The model has been in production for 6 months. New regulations were just introduced that require anonymizing personally identifiable information (PII), which you have identified in your feature set using the Cloud Data Loss Prevention API. You want to update your model pipeline to adhere to the new regulations while minimizing a reduction in model performance. What should you do?",
            "options": [
            "A. Redact the features containing PII data, and train the model from scratch.",
            "B. Mask the features containing PII data, and tune the model from the last checkpoint.",
            "C. Use key-based hashes to tokenize the features containing PII data, and train the model from scratch.",
            "D. Use deterministic encryption to tokenize the features containing PII data, and tune the model from the last checkpoint."
            ],
            "answer": ["C. Use key-based hashes to tokenize the features containing PII data, and train the model from scratch."]
        },
        {
            "question": "You need to train an object detection model to identify bounding boxes around Post-it Notes® in an image. Post-it Notes can have a variety of background colors and shapes. You have a dataset with 1000 images with a maximum size of 1.4MB and a CSV file containing annotations stored in Cloud Storage. You want to select a training method that reliably detects Post-it Notes of any relative size in the image and that minimizes the time to train a model. What should you do?",
            "options": [
            "A. Use the Cloud Vision API in Vertex AI with OBJECT_LOCALIZATION type, and filter the detected objects that match the Post-it Note category only.",
            "B. Upload your dataset into Vertex AI. Use Vertex AI AutoML Vision Object Detection with accuracy as the optimization metric, early stopping enabled, and no training budget specified.",
            "C. Write a Python training application that trains a custom vision model on the training set. Autopackage the application, and configure a custom training job in Vertex AI.",
            "D. Write a Python training application that performs transfer learning on a pre-trained neural network. Autopackage the application, and configure a custom training job in Vertex AI."
            ],
            "answer": ["B. Upload your dataset into Vertex AI. Use Vertex AI AutoML Vision Object Detection with accuracy as the optimization metric, early stopping enabled, and no training budget specified."]
        },
        {
            "question": "You used Vertex AI Workbench notebooks to build a model in TensorFlow. The notebook i) loads data from Cloud Storage, ii) uses TensorFlow Transform to pre-process data, iii) uses built-in TensorFlow operators to define a sequential Keras model, iv) trains and evaluates the model with model.fit() on the notebook instance, and v) saves the trained model to Cloud Storage for serving. You want to orchestrate the model retraining pipeline to run on a weekly schedule while minimizing cost and implementation effort. What should you do?",
            "options": [
            "A. Add relevant parameters to the notebook cells and set a recurring run in Vertex AI Workbench.",
            "B. Use TensorFlow Extended (TFX) with Google Cloud executors to define your pipeline, and automate the pipeline to run on Cloud Composer.",
            "C. Use Kubeflow Pipelines SDK with Google Cloud executors to define your pipeline, and use Vertex AI pipelines to automate the pipeline to run.",
            "D. Separate each cell in the notebook into a containerised application and use Cloud Workflows to launch each application."
            ],
            "answer": ["C. Use Kubeflow Pipelines SDK with Google Cloud executors to define your pipeline, and use Vertex AI pipelines to automate the pipeline to run."]
        },
       {
            "question": "You need to develop an online model prediction service that accesses pre-computed near-real-time features and returns a customer churn probability value. The features are saved in BigQuery and updated hourly using a scheduled query. You want this service to be low latency and scalable and require minimal maintenance. What should you do?",
            "options": [
            "A. 1. Configure Vertex AI Feature Store to automatically import features from BigQuery, and serve them to the model. 2. Deploy the prediction model as a custom Vertex AI endpoint, and enable automatic scaling.",
            "B. 1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Memorystore and performs inference with an in-memory model.",
            "C. 1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use the online service API from Vertex AI Feature Store to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.",
            "D. 1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Vertex AI Feature Store’s online serving API and performs inference with an in-memory model."
            ],
            "answer": ["A. 1. Configure Vertex AI Feature Store to automatically import features from BigQuery, and serve them to the model. 2. Deploy the prediction model as a custom Vertex AI endpoint, and enable automatic scaling."]
        },
        {
            "question": "You are logged into the Vertex AI Pipeline UI and noticed that an automated production TensorFlow training pipeline finished three hours earlier than a typical run. You do not have access to production data for security reasons, but you have verified that no alert was logged in any of the ML system’s monitoring systems and that the pipeline code has not been updated recently. You want to assure the quality of the pipeline results as quickly as possible so you can determine whether to deploy the trained model. What should you do?",
            "options": [
            "A. Use Vertex AI TensorBoard to check whether the training metrics converge to typical values. Verify pipeline input configuration and steps have the expected values.",
            "B. Upgrade to the latest version of the Vertex SDK and re-run the pipeline.",
            "C. Determine the trained model’s location from the pipeline’s metadata in Vertex ML Metadata, and compare the trained model’s size to the previous model.",
            "D. Request access to production systems. Get the training data’s location from the pipeline’s metadata in Vertex ML Metadata, and compare data volumes of the current run to the previous run."
            ],
            "answer": ["A. Use Vertex AI TensorBoard to check whether the training metrics converge to typical values. Verify pipeline input configuration and steps have the expected values."]
        },
        {
            "question": "You recently developed a custom ML model that was trained in Vertex AI on a post-processed training dataset stored in BigQuery. You used a Cloud Run container to deploy the prediction service. The service performs feature lookup and pre-processing and sends a prediction request to a model endpoint in Vertex AI. You want to configure a comprehensive monitoring solution for training-serving skew that requires minimal maintenance. What should you do?",
            "options": [
            "A. Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses email to send alerts. When an alert is received, use the console to diagnose the issue.",
            "B. Update the model hosted in Vertex AI to enable request-response logging. Create a Data Studio dashboard that compares training data and logged data for potential training-serving skew and uses email to send a daily scheduled report.",
            "C. Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged.",
            "D. Update the model hosted in Vertex AI to enable request-response logging. Schedule a daily DataFlow Flex job that uses Tensorflow Data Validation to detect training-serving skew and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged."
            ],
            "answer": ["A. Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses email to send alerts. When an alert is received, use the console to diagnose the issue."]
        },
        {
            "question": "You recently developed a classification model that predicts which customers will be repeat customers. Before deploying the model, you perform post-training analysis on multiple data slices and discover that the model is under-predicting for users who are more than 60 years old. You want to remove age bias while maintaining similar offline performance. What should you do?",
            "options": [
            "A. Perform correlation analysis on the training feature set against the age column, and remove features that are highly correlated with age from the training and evaluation sets.",
            "B. Review the data distribution for each feature against the bucketized age column for the training and evaluation sets, and introduce preprocessing to even irregular feature distributions.",
            "C. Configure the model to support explainability, and modify the input-baselines to include min and max age ranges.",
            "D. Apply a calibration layer at post-processing that matches the prediction distributions of users below and above 60 years old."
            ],
            "answer": ["B. Review the data distribution for each feature against the bucketized age column for the training and evaluation sets, and introduce preprocessing to even irregular feature distributions."]
        },
        {
            "question": "You downloaded a TensorFlow language model pre-trained on a proprietary dataset by another company, and you tuned the model with Vertex AI Training by replacing the last layer with a custom dense layer. The model achieves the expected offline accuracy; however, it exceeds the required online prediction latency by 20ms. You want to reduce latency while minimizing the offline performance drop and modifications to the model before deploying the model to production. What should you do?",
            "options": [
            "A. Apply post-training quantization on the tuned model, and serve the quantized model.",
            "B. Apply knowledge distillation to train a new, smaller \"student\" model that mimics the behavior of the larger, fine-tuned model.",
            "C. Use pruning to tune the pre-trained model on your dataset, and serve the pruned model after stripping it of training variables.",
            "D. Use clustering to tune the pre-trained model on your dataset, and serve the clustered model after stripping it of training variables."
            ],
            "answer": ["A. Apply post-training quantization on the tuned model, and serve the quantized model."]
        },
        {
            "question": "You have a dataset that is split into training, validation, and test sets. All the sets have similar distributions. You have sub-selected the most relevant features and trained a neural network. TensorBoard plots show the training loss oscillating around 0.9, with the validation loss higher than the training loss by 0.3. You want to update the training regime to maximize the convergence of both losses and reduce overfitting. What should you do?",
            "options": [
            "A. Decrease the learning rate to fix the validation loss, and increase the number of training epochs to improve the convergence of both losses.",
            "B. Decrease the learning rate to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses.",
            "C. Introduce L1 regularization to fix the validation loss, and increase the learning rate and the number of training epochs to improve the convergence of both losses.",
            "D. Introduce L2 regularization to fix the validation loss."
            ],
            "answer": ["D. Introduce L2 regularization to fix the validation loss."]
        },
        {
            "question": "You recently used Vertex AI Prediction to deploy a custom-trained model in production. The automated re-training pipeline made available a new model version that passed all unit and infrastructure tests. You want to define a rollout strategy for the new model version that guarantees an optimal user experience with zero downtime. What should you do?",
            "options": [
            "A. Release the new model version in the same Vertex AI endpoint. Use traffic splitting in Vertex AI Prediction to route a small random subset of requests to the new version and, if the new version is successful, gradually route the remaining traffic to it.",
            "B. Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application.",
            "C. Deploy the current model version with an Istio resource in Google Kubernetes Engine, and route production traffic to it. Deploy the new model version, and use Istio to route a small random subset of traffic to it. If the new version is successful, gradually route the remaining traffic to it.",
            "D. Install Seldon Core and deploy an Istio resource in Google Kubernetes Engine. Deploy the current model version and the new model version using the multi-armed bandit algorithm in Seldon to dynamically route requests between the two versions before eventually routing all traffic over to the best-performing version."
            ],
            "answer": ["B. Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application."]
        },
        {
            "question": "You work as an analyst at a large banking firm. You are developing a robust, scalable ML pipeline to train several regression and classification models. Your primary focus for the pipeline is model interpretability. You want to productionize the pipeline as quickly as possible. What should you do?",
            "options": [
            "A. Use Tabular Workflow for Wide & Deep through Vertex AI Pipelines to jointly train wide linear models and deep neural networks.",
            "B. Use Cloud Composer to build the training pipelines for custom deep learning-based models.",
            "C. Use Google Kubernetes Engine to build a custom training pipeline for XGBoost-based models.",
            "D. Use Tabular Workflow for TabNet through Vertex AI Pipelines to train attention-based models."
            ],
            "answer": ["D. Use Tabular Workflow for TabNet through Vertex AI Pipelines to train attention-based models."]
        },
        {
            "question": "You are developing a custom image classification model in Python. You plan to run your training application on Vertex AI. Your input dataset contains several hundred thousand small images. You need to determine how to store and access the images for training. You want to maximize data throughput and minimize training time while reducing the amount of additional code. What should you do?",
            "options": [
            "A. Store image files in Cloud Storage, and access them directly.",
            "B. Store image files in Cloud Storage, and access them by using serialized records.",
            "C. Store image files in Cloud Filestore, and access them by using serialized records.",
            "D. Store image files in Cloud Filestore, and access them directly by using an NFS mount point."
            ],
            "answer": ["C. Store image files in Cloud Filestore, and access them by using serialized records."]
        },
        { 
            "question": "Your company manages an ecommerce website. You developed an ML model that recommends additional products to users in near real time based on items currently in the user’s cart. The workflow will include the following processes:\n\n1. The website will send a Pub/Sub message with the relevant data, and then receive a message with the prediction from Pub/Sub. \n2. Predictions will be stored in BigQuery. \n3. The model will be stored in a Cloud Storage bucket and will be updated frequently.\n\nYou want to minimize prediction latency and the effort required to update the model. How should you reconfigure the architecture?",
            "options": [
            "A. Write a Cloud Function that loads the model into memory for prediction. Configure the function to be triggered when messages are sent to Pub/Sub.",
            "B. Expose the model as a Vertex AI endpoint. Write a custom DoFn in a Dataflow job that calls the endpoint for prediction.",
            "C. Use the RunInference API with WatchFilePattern in a Dataflow job that wraps around the model and serves predictions.",
            "D. Create a pipeline in Vertex AI Pipelines that performs preprocessing, prediction, and postprocessing. Configure the pipeline to be triggered by a Cloud Function when messages are sent to Pub/Sub."
            ],
            "answer": ["C. Use the RunInference API with WatchFilePattern in a Dataflow job that wraps around the model and serves predictions."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def assessment():
    st.title("Assessment Test")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    questions_assessment = [
        {
            "question": "How would you split the data to predict a user lifetime value (LTV) over the next 30 days in an online recommendation system to avoid data and label leakage? (Choose three.)",
            "options": [
            "A. Perform data collection for 30 days.",
            "B. Create a training set for data from day 1 to day 29.",
            "C. Create a validation set for data for day 30.",
            "D. Create random data split into training, validation, and test sets."
            ],
            "answer": [
            "A. Perform data collection for 30 days.",
            "B. Create a training set for data from day 1 to day 29.",
            "C. Create a validation set for data for day 30."
            ]
        },
        {
            "question": "You have a highly imbalanced dataset and you want to focus on the positive class in the classification problem. Which metrics would you choose?",
            "options": [
            "A. Area under the precision‐recall curve (AUC PR)",
            "B. Area under the curve ROC (AUC ROC)",
            "C. Recall",
            "D. Precision"
            ],
            "answer": "A. Area under the precision‐recall curve (AUC PR)"
        },
        {
            "question": "A feature cross is created by ________________ two or more features.",
            "options": [
            "A. Swapping",
            "B. Multiplying",
            "C. Adding",
            "D. Dividing"
            ],
            "answer": "B. Multiplying"
        },
        {
            "question": "You can use Cloud Pub/Sub to stream data in GCP and use Cloud Dataflow to transform the data.",
            "options": [
            "A. True",
            "B. False"
            ],
            "answer": "A. True"
        },
        {
            "question": "You have training data, and you are writing the model training code. You have a team of data engineers who prefer to code in SQL. Which service would you recommend?",
            "options": [
            "A. BigQuery ML",
            "B. Vertex AI custom training",
            "C. Vertex AI AutoML",
            "D. Vertex AI pretrained APIs"
            ],
            "answer": "A. BigQuery ML"
        },
        {
            "question": "What are the benefits of using a Vertex AI managed dataset? (Choose three.)",
            "options": [
            "A. Integrated data labeling for unlabeled, unstructured data such as video, text, and images using Vertex data labeling.",
            "B. Track lineage to models for governance and iterative development.",
            "C. Automatically splitting data into training, test, and validation sets.",
            "D. Manual splitting of data into training, test, and validation sets."
            ],
            "answer": [
            "A. Integrated data labeling for unlabeled, unstructured data such as video, text, and images using Vertex data labeling.",
            "B. Track lineage to models for governance and iterative development.",
            "C. Automatically splitting data into training, test, and validation sets."
            ]
        },
        {
            "question": "Masking, encrypting, and bucketing are de‐identification techniques to obscure PII data using the Cloud Data Loss Prevention API.",
            "options": [
            "A. True",
            "B. False"
            ],
            "answer": "A. True"
        },
        {
            "question": "Which strategy would you choose to handle the sensitive data that exists within images, videos, audio, and unstructured freeform data?",
            "options": [
            "A. Use NLP API, Cloud Speech API, Vision AI, and Video Intelligence AI to identify sensitive data such as email and location out of box, and then redact or remove it.",
            "B. Use Cloud DLP to address this type of data.",
            "C. Use Healthcare API to hide sensitive data.",
            "D. Create a view that doesn't provide access to the columns in question. The data engineers cannot view the data, but at the same time the data is live and doesn't require human intervention to de‐identify it for continuous training."
            ],
            "answer": "A. Use NLP API, Cloud Speech API, Vision AI, and Video Intelligence AI to identify sensitive data such as email and location out of box, and then redact or remove it."
        },
        {
            "question": "You would use __________________ when you are trying to reduce features while trying to solve an overfitting problem with large models.",
            "options": [
            "A. L1 regularization",
            "B. L2 regularization",
            "C. Both A and B",
            "D. Vanishing gradient"
            ],
            "answer": "A. L1 regularization"
        },
        {
            "question": "If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms leading to exploding gradients that get too large to converge. What are some of the ways this can be avoided? (Choose two.)",
            "options": [
            "A. Batch normalization",
            "B. Lower learning rate",
            "C. The ReLU activation function",
            "D. Sigmoid activation function"
            ],
            "answer": [
            "A. Batch normalization",
            "B. Lower learning rate"
            ]
        },
        {
            "question": "You have a Spark and Hadoop environment on‐premises, and you are planning to move your data to Google Cloud. Your ingestion pipeline is both real time and batch. Your ML customer engineer recommended a scalable way to move your data using Cloud Dataproc to BigQuery. Which of the following Dataproc connectors would you *not* recommend?",
            "options": [
            "A. Pub/Sub Lite Spark connector",
            "B. BigQuery Spark connector",
            "C. BigQuery connector",
            "D. Cloud Storage connector"
            ],
            "answer": "D. Cloud Storage connector"
        },
        {
            "question": "You have moved your Spark and Hadoop environment and your data is in Google Cloud Storage. Your ingestion pipeline is both real time and batch. Your ML customer engineer recommended a scalable way to run Apache Hadoop or Apache Spark jobs directly on data in Google Cloud Storage. Which of the following Dataproc connector would you recommend?",
            "options": [
            "A. Pub/Sub Lite Spark connector",
            "B. BigQuery Spark connector",
            "C. BigQuery connector",
            "D. Cloud Storage connector"
            ],
            "answer": "D. Cloud Storage connector"
        },
        {
            "question": "Which of the following is *not* a technique to speed up hyperparameter optimization?",
            "options": [
            "A. Parallelize the problem across multiple machines by using distributed training with hyperparameter optimization.",
            "B. Avoid redundant computations by pre‐computing or cache the results of computations that can be reused for subsequent model fits.",
            "C. Use grid search rather than random search.",
            "D. If you have a large dataset, use a simple validation set instead of cross‐validation."
            ],
            "answer": "C. Use grid search rather than random search."
        },
        {
            "question": "Vertex AI Vizier is an independent service for optimizing complex models with many parameters. It can be used only for non‐ML use cases.",
            "options": [
            "A. True",
            "B. False"
            ],
            "answer": "B. False"
        },
        {
            "question": "Which of the following is *not* a tool to track metrics when training a neural network?",
            "options": [
            "A. Vertex AI interactive shell",
            "B. What‐If Tool",
            "C. Vertex AI TensorBoard Profiler",
            "D. Vertex AI hyperparameter tuning"
            ],
            "answer": "D. Vertex AI hyperparameter tuning"
        },
        {
            "question": "You are a data scientist working to select features with structured datasets. Which of the following techniques will help?",
            "options": [
            "A. Sampled Shapley",
            "B. Integrated gradient",
            "C. XRAI (eXplanation with Ranked Area Integrals)",
            "D. Gradient descent"
            ],
            "answer": "A. Sampled Shapley"
        },
        {
            "question": "Variable selection and avoiding target leakage are the benefits of feature importance.",
            "options": [
            "A. True",
            "B. False"
            ],
            "answer": "A. True"
        },
        {
            "question": "A TensorFlow SavedModel is what you get when you call __________________. Saved models are stored as a directory on disk. The file within that directory, saved_model.pb, is a protocol buffer describing the functional tf.Graph.",
            "options": [
            "A. tf.saved_model.save()",
            "B. tf.Variables",
            "C. tf.predict()",
            "D. Tf.keras.models.load_model"
            ],
            "answer": "A. tf.saved_model.save()"
        },
        {
            "question": "What steps would you recommend a data engineer trying to deploy a TensorFlow model trained locally to set up real‐time prediction using Vertex AI? (Choose three.)",
            "options": [
            "A. Import the model to Model Registry.",
            "B. Deploy the model.",
            "C. Create an endpoint for deployed model.",
            "D. Create a model in Model Registry."
            ],
            "answer": [
            "A. Import the model to Model Registry.",
            "B. Deploy the model.",
            "C. Create an endpoint for deployed model."
            ]
        },
        {
            "question": "You are an MLOps engineer and you deployed a Kubeflow pipeline on Vertex AI pipelines. Which Google Cloud feature will help you track lineage with your Vertex AI pipelines?",
            "options": [
            "A. Vertex AI Model Registry",
            "B. Vertex AI Artifact Registry",
            "C. Vertex AI ML metadata",
            "D. Vertex AI Model Monitoring"
            ],
            "answer": "C. Vertex AI ML metadata"
        },
        {
            "question": "What is not a recommended way to invoke a Kubeflow pipeline?",
            "options": [
                "A. Using Cloud Scheduler",
                "B. Responding to an event, using Pub/Sub and Cloud Functions",
                "C. Cloud Composer and Cloud Build",
                "D. Directly using BigQuery"
            ],
            "answer": [
                "D. Directly using BigQuery"
            ]
        },
        {
            "question": "You are a software engineer working at a start-up that works on organizing personal photos and pet photos. You have been asked to use machine learning to identify and tag which photos have pets and also identify public landmarks in the photos. These features are not available today and you have a week to create a solution for this. What is the best approach?",
            "options": [
                "A. Find the best cat/dog dataset and train a custom model on Vertex AI using the latest algorithm available. Do the same for identifying landmarks.",
                "B. Find a pretrained cat/dog dataset (available) and train a custom model on Vertex AI using the latest deep neural network TensorFlow algorithm.",
                "C. Use the cat/dog dataset to train a Vertex AI AutoML image classification model on Vertex AI. Do the same for identifying landmarks.",
                "D. Vision AI already identifies pets and landmarks. Use that to see if it meets the requirements. If not, use the Vertex AI AutoML model."
            ],
            "answer": [
                "D. Vision AI already identifies pets and landmarks. Use that to see if it meets the requirements. If not, use the Vertex AI AutoML model."
            ]
        },
        {
            "question": "You are building a product that will accurately throw a ball into the basketball net. This should work no matter where it is placed on the court. You have created a very large TensorFlow model (size more than 90 GB) based on thousands of hours of video. The model uses custom operations, and it has optimized the training loop to not have any I/O operations. What are your hardware options to train this model?",
            "options": [
                "A. Use a TPU slice because the model is very large and has been optimized to not have any I/O operations.",
                "B. Use a TPU pod because the model size is larger than 50 GB.",
                "C. Use a GPU-only instance.",
                "D. Use a CPU-only instance to build your model."
            ],
            "answer": [
                "C. Use a GPU-only instance."
            ]
        },
        {
            "question": "You work in the fishing industry and have been asked to use machine learning to predict the age of lobster based on size and color. You have thousands of images of lobster from Arctic fishing boats, from which you have extracted the size of the lobster that is passed to the model, and you have built a regression model for predicting age. Your model has performed very well in your test and validation data. Users want to use this model from their boats. What are your next steps? (Choose three.)",
            "options": [
                "A. Deploy the model on Vertex AI, expose a REST endpoint.",
                "B. Enable monitoring on the endpoint and see if there is any training-serving skew and drift detection. The original dataset was only from Arctic boats.",
                "C. Also port this model to BigQuery for batch prediction.",
                "D. Enable Vertex AI logging and analyze the data in BigQuery."
            ],
            "answer": [
                "A. Deploy the model on Vertex AI, expose a REST endpoint.",
                "B. Enable monitoring on the endpoint and see if there is any training-serving skew and drift detection. The original dataset was only from Arctic boats.",
                "D. Enable Vertex AI logging and analyze the data in BigQuery."
            ]
        },
        {
            "question": "You have built a custom model and deployed it in Vertex AI. You are not sure if the predictions are being served fast enough (low latency). You want to measure this by enabling Vertex AI logging. Which type of logging will give you information like time stamp and latency for each request?",
            "options": [
                "A. Container logging",
                "B. Time stamp logging",
                "C. Access logging",
                "D. Request-response logging"
            ],
            "answer": [
                "C. Access logging"
            ]
        },
        {
            "question": "You are part of a growing ML team in your company that has started to use machine learning to improve your business. You were initially building models using Vertex AI AutoML and providing the trained models to the deployment teams. How should you scale this?",
            "options": [
                "A. Create a Python script to train multiple models using Vertex AI.",
                "B. You are now in level 0, and your organization needs level 1 MLOps maturity. Automate the training using Vertex AI Pipelines.",
                "C. You are in the growth phase of the organization, so it is important to grow the team to leverage more ML engineers.",
                "D. Move to Vertex AI custom models to match the MLOps maturity level."
            ],
            "answer": "B. You are now in level 0, and your organization needs level 1 MLOps maturity. Automate the training using Vertex AI Pipelines."
        },
        {
            "question": "What is not a reason to use Vertex AI Feature Store?",
            "options": [
                "A. It is a managed service.",
                "B. It extracts features from images and videos and stores them.",
                "C. All data is a time‐series, so you can track when the features values change over time.",
                "D. The features created by the feature engineering teams are available during training time but not during serving time. So this helps in bridging that."
            ],
            "answer": "B. It extracts features from images and videos and stores them."
        },
        {
            "question": "You are a data analyst in an organization that has thousands of insurance agents, and you have been asked to predict the revenue by each agent for the next quarter. You have the historical data for the last 10 years. You are familiar with all AI services on Google Cloud. What is the most efficient way to do this?",
            "options": [
                "A. Build a Vertex AI AutoML forecast, deploy the model, and make predictions using REST API.",
                "B. Build a Vertex AI AutoML forecast model, import the model into BigQuery, and make predictions using BigQuery ML.",
                "C. Build a BigQuery ML ARIMA+ model using data in BigQuery, and make predictions in BigQuery.",
                "D. Build a BigQuery ML forecast model, export the model to Vertex AI, and run a batch prediction in Vertex AI."
            ],
            "answer": "C. Build a BigQuery ML ARIMA+ model using data in BigQuery, and make predictions in BigQuery."
        },
        {
            "question": "You are an expert in Vertex AI Pipelines, Vertex AI training, and Vertex AI deployment and monitoring. A data analyst team has built a highly accurate model, and this has been brought to you. Your manager wants you to make predictions using the model and use those predictions. What do you do?",
            "options": [
                "A. Retrain the model on Vertex AI with the same data and deploy the model on Vertex AI as part of your CD.",
                "B. Run predictions on BigQuery ML and export the predictions into GCS and then load into your pipeline.",
                "C. Export the model from BigQuery into the Vertex AI model repository and run predictions in Vertex AI.",
                "D. Download the BigQuery model, and package into a Vertex AI custom container and deploy it in Vertex AI."
            ],
            "answer": "C. Export the model from BigQuery into the Vertex AI model repository and run predictions in Vertex AI."
        },
        {
            "question": "Which of the following statements about Vertex AI and BigQuery ML is incorrect?",
            "options": [
                "A. BigQueryML supports both unsupervised and supervised models.",
                "B. BigQuery ML is very portable. Vertex AI supports all models trained on BigQuery ML.",
                "C. Vertex AI model monitoring and logs data is stored in BigQuery tables.",
                "D. BigQuery ML also has algorithms to predict recommendations for users."
            ],
            "answer": "B. BigQuery ML is very portable. Vertex AI supports all models trained on BigQuery ML."
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions_assessment):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap1():
    st.title("Chapter 1: Framing ML Problems")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Translate business challenges to machine learning**. Understand the business use case that wants to solve a problem using machine learning. Understand the type of problem, the data availability, expected outcomes, stakeholders, budget, and timelines." "\n- **Understand the problem types**. Understand regression, classification, and forecasting. Be able to tell the difference in data types and popular algorithms for each problem type." "\n- **Know how to use ML metrics**. Understand what a metric is, and match the metric with the use case. Know the different metrics for each problem type, like precision, recall, F1, AUC ROC, RMSE, and MAPE." "\n- **Understand Google's Responsible AI principles**. Understand the recommended practices for AI in the context of fairness, interpretability, privacy, and security."
    questions_chap1 = [
        {
            "question": "When analyzing a potential use case, what are the first things you should look for? (Chose three.)",
            "options": [
                "A. Impact",
                "B. Success criteria",
                "C. Algorithm",
                "D. Budget and time frames"
            ],
            "answer": ["A. Impact", "B. Success criteria", "D. Budget and time frames"]
        },
        {
            "question": "When you try to find the best ML problem for a business use case, which of these aspects is *not* considered?",
            "options": [
                "A. Model algorithm",
                "B. Hyperparameters",
                "C. Metric",
                "D. Data availability"
            ],
            "answer": "B. Hyperparameters"  # Corrected answer based on context
        },
        {
            "question": "Your company wants to predict the amount of rainfall for the next 7 days using machine learning. What kind of ML problem is this?",
            "options": [
                "A. Classification",
                "B. Forecasting",
                "C. Clustering",
                "D. Reinforcement learning"
            ],
            "answer": "B. Forecasting"
        },
        {
            "question": "You work for a large company that gets thousands of support tickets daily. Your manager wants you to create a machine learning model to detect if a support ticket is valid or not. What type of model would you choose?",
            "options": [
                "A. Linear regression",
                "B. Binary classification",
                "C. Topic modeling",
                "D. Multiclass classification"
            ],
            "answer": "B. Binary classification"
        },
        {
            "question": "You are building an advanced camera product for sports, and you want to track the ball. What kind of problem is this?",
            "options": [
                "A. Not possible with current state‐of‐the‐art algorithms",
                "B. Image detection",
                "C. Video object tracking",
                "D. Scene detection"
            ],
            "answer": "C. Video object tracking"
        },
        {
            "question": "Your company has millions of academic papers from several research teams. You want to organize them in some way, but there is no company policy on how to classify the documents. You are looking for any way to cluster the documents and gain any insight into popular trends. What can you do?",
            "options": [
                "A. Not much. The problem is not well defined.",
                "B. Use a simple regression problem.",
                "C. Use binary classification.",
                "D. Use topic modeling."
            ],
            "answer": "D. Use topic modeling."
        },
        {
            "question": "What metric would you never chose for linear regression?",
            "options": [
                "A. RMSE",
                "B. MAPE",
                "C. Precision",
                "D. MAE"
            ],
            "answer": "C. Precision"
        },
        {
            "question": "You are building a machine learning model to predict house prices. You want to make sure the prediction does not have extreme errors. What metric would you choose?",
            "options": [
                "A. RMSE",
                "B. RMSLE",
                "C. MAE",
                "D. MAPE"
            ],
            "answer": "A. RMSE"
        },
        {
            "question": "You are building a plant classification model to predict variety1 and variety2, which are found in equal numbers in the field. What metric would you choose?",
            "options": [
                "A. Accuracy",
                "B. RMSE",
                "C. MAPE",
                "D. R2"
            ],
            "answer": "A. Accuracy"
        },
        {
            "question": "You work for a large car manufacturer and are asked to detect hidden cracks in engines using X‐ray images. However, missing a crack could mean the engine could fail at some random time while someone is driving the car. Cracks are relatively rare and happen in about 1 in 100 engines. A special camera takes an X-ray image of the engine as it comes through the assembly line. You are going to build a machine learning model to classify if an engine has a crack or not. If a crack is detected, the engine would go through further testing to verify. What metric would you choose for your classification model?",
            "options": [
                "A. Accuracy",
                "B. Precision",
                "C. Recall",
                "D. RMSE"
            ],
            "answer": "C. Recall"
        },
        {
            "question": "You are asked to build a classification model and are given a training dataset but the data is not labeled. You are asked to identify ways of using machine learning with this data. What type of learning will you use?",
            "options": [
                "A. Supervised learning",
                "B. Unsupervised learning",
                "C. Semi‐supervised learning",
                "D. Reinforcement learning"
            ],
            "answer": "B. Unsupervised learning"
        },
        {
            "question": "You work at a company that hosts millions of videos and you have thousands of users. The website has a Like button for users to click, and some videos get thousands of “likes.” You are asked to create a machine learning model to recommend videos to users based on all the data collected to increase the amount of time users spend on your website. What would be your ML approach?",
            "options": [
                "A. Supervised learning to predict based on the popularity of videos",
                "B. Deep learning model based on the amount of time users watch the videos",
                "C. Collaborative filtering method based on explicit feedback",
                "D. Semi-supervised learning because you have some data about some videos"
            ],
            "answer": "C. Collaborative filtering method based on explicit feedback"
        },
        {
            "question": "You work for the web department of a large hardware store chain. You have built a visual search engine for the website. You want to build a model to classify whether an image contains a product. There are new products being introduced on a weekly basis to your product catalog and these new products must be incorporated into the visual search engine. Which of the following options is a *bad* idea?",
            "options": [
                "A. Create a pipeline to automate the step: take the dataset, train a model.",
                "B. Create a golden dataset and do not change the dataset for at least a year because creating a dataset is time-consuming.",
                "C. Extend the dataset to include new products frequently and retrain the model.",
                "D. Add evaluation of the model as part of the pipeline."
            ],
            "answer": "B. Create a golden dataset and do not change the dataset for at least a year because creating a dataset is time-consuming."
        },
        {
            "question": "Which of the following options is not a type of machine learning approach?",
            "options": [
                "A. Supervised learning",
                "B. Unsupervised learning",
                "C. Semi-supervised learning",
                "D. Hyper-supervised learning"
            ],
            "answer": "D. Hyper-supervised learning"
        },
        {
            "question": "Your manager is discussing a machine learning approach and is asking you about feeding the output of one model to another model. Select two statements that are true about this kind of approach.",
            "options": [
                "A. There are many ML pipelines where the output of one model is fed into another.",
                "B. This is a poor design and never done in practice.",
                "C. Never feed the output of one model into another model. It may amplify errors.",
                "D. There are several design patterns where the output of one model (like encoder or transformer) is passed into a second model and so on."
            ],
            "answer": ["A. There are many ML pipelines where the output of one model is fed into another.", "D. There are several design patterns where the output of one model (like encoder or transformer) is passed into a second model and so on."]
        },
        {
            "question": "You are building a model that is going to predict creditworthiness and will be used to approve loans. You have created a model and it is performing extremely well and has high impact. What next?",
            "options": [
                "A. Deploy the model.",
                "B. Deploy the model and integrate it with the system.",
                "C. Hand it over to the software integration team.",
                "D. Test your model and data for biases (gender, race, etc.)."
            ],
            "answer": "D. Test your model and data for biases (gender, race, etc.)."
        },
        {
            "question": "You built a model to predict credit‐worthiness, and your training data was checked for biases. Your manager still wants to know the reason for each prediction and what the model does. What do you do?",
            "options": [
                "A. Get more testing data.",
                "B. The ML model is a black box. You cannot satisfy this requirement.",
                "C. Use model interpretability/explanations.",
                "D. Remove all fields that may cause bias (race, gender, etc.)."
            ],
            "answer": "C. Use model interpretability/explanations."
        },
        {
            "question": "Your company is building an Android app to add funny moustaches on photos. You built a deep learning model to detect the location of a face in a photo, and your model had very high accuracy based on a public photo dataset that you found online. When integrated into an Android phone app, it got negative feedback on accuracy. What could be the reason?",
            "options": [
                "A. The model was not deployed properly.",
                "B. Android phones could not handle a deep learning model.",
                "C. Your dataset was not representative of all users.",
                "D. The metric was wrong."
            ],
            "answer": "C. Your dataset was not representative of all users."
        },
        {
            "question": "You built a deep learning model to predict cancer based on thousands of personal records and scans. The data was used in training and testing. The model is secured behind a firewall, and all cybersecurity precautions have been taken. Are there any privacy concerns? (Chose two.)",
            "options": [
                "A. No. There are no privacy concerns. This does not contain photographs, only scans.",
                "B. Yes. This is sensitive data being used.",
                "C. No. Although sensitive data is used, it is only for training and testing.",
                "D. The model could reveal some detail about the training data. There is a risk."
            ],
            "answer": ["B. Yes. This is sensitive data being used.", "D. The model could reveal some detail about the training data. There is a risk."]
        },
        {
            "question": "You work for an online shoe store and the company wants to increase revenue. You have a large dataset that includes the browsing history of thousands of customers, and also their shopping cart history. You have been asked to create a recommendation model. Which of the following is *not* a valid next step?",
            "options": [
                "A. Use your ML model to recommend products at checkout.",
                "B. Creatively use all the data to get maximum value because there is no privacy concern.",
                "C. Periodically retrain the model to adjust for performance and also to include new products.",
                "D. In addition to the user history, you can use the data about product (description, images) in training your model."
            ],
            "answer": "B. Creatively use all the data to get maximum value because there is no privacy concern."
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions_chap1):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap2():
    st.title("Chapter 2: Exploring Data and Building Data Pipelines")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Be able to visualize data**. Understand why we need to visualize data and various ways to do so, such as using box plots, line plots, and scatterplots.""\n- **Understand the fundamentals of statistical terms**. Be able to describe mean, median, mode, and standard deviation and how they are relevant in finding outliers in data. Also know how to check data correlation using a line plot.""\n- **Determine data quality and reliability or feasibility**. Understand why you want data without outliers and what data skew is, and learn about various data cleaning and normalizing techniques such as log scaling, scaling, clipping, and z-score.""\n- **Establish data constraints**. Understand why it's important to define a data schema in an ML pipeline and the need to validate data. Also, you need to understand TFDV for validating data at scale.""\n- **Organize and optimize training data**. You need to understand how to split your dataset into training data, test data, and validation data and how to apply the data splitting technique when you have clustered and online data. Also understand the sampling strategy when you have imbalanced data.""\n- **Handle missing data**. Know the various ways to handle missing data, such as removing missing values; replacing missing values with mean, median, or mode; or using ML to create missing values.""\n- **Avoid data leaks**. Know the various ways data leakage and label leakage can happen in the data and how to avoid it."
    
    questions_chap2 = [
        {
            "question": "You are the data scientist for your company. You have a dataset that includes credit card transactions, and 1 percent of those credit card transactions are fraudulent. Which data transformation strategy would likely improve the performance of your classification model?",
            "options": [
                "A. Write your data in TFRecords.",
                "B. Z‐normalize all the numeric features.",
                "C. Use one‐hot encoding on all categorical features.",
                "D. Oversample the fraudulent transactions."
            ],
            "answer": "D. Oversample the fraudulent transactions."
        },
        {
            "question": "You are a research scientist building a cancer prediction model from medical records. Features of the model are patient name, hospital name, age, vitals, and test results. This model performed really well on held‐out test data but performed poorly on new patient data. What is the reason for this?",
            "options": [
                "A. Strong correlation between feature hospital name and predicted result.",
                "B. Random splitting of data between all the features available.",
                "C. Missing values in the feature hospital name and age.",
                "D. Negative correlation between the feature hospital name and age."
            ],
            "answer": "A. Strong correlation between feature hospital name and predicted result."
        },
        {
            "question": "Your team trained and tested a deep neural network model with 99 percent accuracy. Six months after model deployment, the model is performing poorly due to change in input data distribution. How should you address input data distribution?",
            "options": [
                "A. Create alerts to monitor for skew and retrain your model.",
                "B. Perform feature selection and retrain the model.",
                "C. Retrain the model after hyperparameter tuning.",
                "D. Retrain your model monthly to detect data skew."
            ],
            "answer": "A. Create alerts to monitor for skew and retrain your model."
        },
        {
            "question": "You are an ML engineer who builds and manages a production system to predict sales. Model accuracy is important as the production model has to keep up with market changes. After a month in production, the model did not change but the model accuracy was reduced. What is the most likely cause of the reduction in model accuracy?",
            "options": [
                "A. Accuracy dropped due to poor quality data.",
                "B. Lack of model retraining.",
                "C. Incorrect data split ratio in validation, test, and training data.",
                "D. Missing data for training."
            ],
            "answer": "B. Lack of model retraining."  
        },
        {
            "question": "You are a data scientist in a manufacturing firm. You have been asked to investigate failure of a production line based on sensor readings. You realize that 1 percent of the data samples are positive examples of a faulty sensor reading. How will you resolve the class imbalance problem?",
            "options": [
                "A. Generate 10 percent positive examples using class distribution.",
                "B. Downsample the majority data with upweighting to create 10 percent samples.",
                "C. Delete negative examples until positive and negative examples are equal.",
                "D. Use a convolutional neural network with the softmax activation function."
            ],
            "answer": "B. Downsample the majority data with upweighting to create 10 percent samples." 
        },
        {
            "question": ":red[[Suspected faulty answer]] You are the data scientist of a meteorological department asked to build a model to predict daily temperatures. You split the data randomly and then transform the training and test datasets. Temperature data for model training is uploaded hourly. During testing, your model performed with 99 percent accuracy; however, in production, accuracy dropped to 70 percent. How can you improve the accuracy of your model in production?",
            "options": [
                "A. Split the training and test data based on time rather than a random split to avoid leakage.",
                "B. Normalize the data for the training and test datasets as two separate steps.",
                "C. Add more data to your dataset so that you have fair distribution.",
                "D. Transform data before splitting, and cross‐validate to make sure the transformations are applied to both the training and test sets."
            ],
            "answer": "D. Transform data before splitting, and cross‐validate to make sure the transformations are applied to both the training and test sets." # Addresses data leakage caused by transformations after splitting.
        },
        {
            "question": ":red[[Suspected faulty answer]]You are working on a neural‐network‐based project. The dataset provided to you has columns with different ranges and a lot of missing values. While preparing the data for model training, you discover that gradient optimization is having difficulty moving weights. What should you do?",
            "options": [
                "A. Use feature construction to combine the strongest features.",
                "B. Use the normalization technique to transform data.",
                "C. Improve the data cleaning step by removing features with missing values.",
                "D. Change the hyperparameter tuning steps to reduce the dimension of the test set and have a larger training set."
            ],
            "answer": "C. Improve the data cleaning step by removing features with missing values."
        },
        {
            "question": "You are an ML engineer working to set a model in production. Your model performs well with training data. However, the model performance degrades in production environment and your model is overfitting. What can be the reason for this? (Choose three.)",
            "options": [
                "A. Applying normalizing features such as removing outliers to the entire dataset",
                "B. High correlation between the target variable and the feature",
                "C. Removing features with missing values",
                "D. Adding your target variable as your feature"
            ],
            "answer": ["A. Applying normalizing features such as removing outliers to the entire dataset", 
                       "B. High correlation between the target variable and the feature",
                       "D. Adding your target variable as your feature"]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions_chap2):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap3():
    st.title("Chapter 3: Feature Engineering")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Use consistent data processing**. Understand when to transform data, either before training or during model training. Also know the benefits and limitations of transforming data before training.""\n- **Know how to encode structured data types**. Understand techniques to transform both numeric and categorical data such as bucketing, normalization, hashing, and one-hot encoding.""\n- **Understand feature selection**. Understand why feature selection is needed and some of the techniques of feature selection, such as dimensionality reduction.""\n- **Understand class imbalance**. Understand true positive, false positive, accuracy, AUC, precision, and recall in classification problems and how to effectively measure accuracy with class imbalance.""\n- **Know where and how to use feature cross**. You need to understand why feature cross is important and the scenarios in which you would need it.""\n- **Understand TensorFlow Transform**. You need to understand TensorFlow Data and TensorFlow Transform and how to architect tf.Transform pipelines on Google Cloud using BigQuery and Cloud Data Fusion.""\n- **Use GCP data and ETL tools**. Know how and when to use tools such as Cloud Data Fusion and Cloud Dataprep. For example, in case you are looking for a no-code solution to clean data, you would use Dataprep for data processing and, in case you are looking for a no-code and UI-based solution for ETL (extract, transform, load), you would use Cloud Data Fusion."
    
    questions_chap3 = [
        {
            "question": "You are the data scientist for your company. You have a dataset that which has all categorical features. You trained a model using some algorithms. With some algorithms this data is giving good result but when you change the algorithm the performance is getting reduced. Which data transformation strategy ould likely improve the performance of your model?",
            "options": [
            "A. Write your data in TFRecords.",
            "B. Create a feature cross with categorical feature.",
            "C. Use one‐hot encoding on all categorical features.",
            "D. Oversample the features."
            ],
            "answer": ["C. Use one‐hot encoding on all categorical features."]
        },
        {
            "question": "You are working on a neural network–based project. The dataset provided to you has columns with different ranges. While preparing the data for model training, you discover that gradient optimization is having difficulty moving weights to an optimized solution. What should you do?",
            "options": [
            "A. Use feature construction to combine the strongest features.",
            "B. Use the normalization technique.",
            "C. Improve the data cleaning step by removing features with missing values.",
            "D. Change the partitioning step to reduce the dimension of the test set and have a larger training set."
            ],
            "answer": ["B. Use the normalization technique."]
        },
        {
            "question": "You work for a credit card company and have been asked to create a custom fraud detection model based on historical data using AutoML Tables. You need to prioritize detection of fraudulent transactions while minimizing false positives.",
            "options": [
            "A. An optimization objective that minimizes log loss.",
            "B. An optimization objective that maximizes the precision at a recall value of 0.50.",
            "C. An optimization objective that maximizes the area under the precision‐recall curve (AUC PR) value.",
            "D. An optimization objective that maximizes the area under the curve receiver operating characteristic (AUC ROC) curve value."
            ],
            "answer": ["C. An optimization objective that maximizes the area under the precision‐recall curve (AUC PR) value."]
        },
        {
            "question": ":red[[Suspected faulty answer]]You are a data scientist working on a classification problem with time‐series data and achieved an area under the receiver operating characteristic curve (AUC ROC) value of 99 percent for training data with just a few experiments. You haven't explored using any sophisticated algorithms or spent any time on hyperparameter tuning. What should your next step be to identify and fix the problem?",
            "options": [
            "A. Address the model overfitting by using a less complex algorithm.",
            "B. Address data leakage by applying nested cross‐validation during model training.",
            "C. Address data leakage by removing features highly correlated with the target value.",
            "D. Address the model overfitting by tuning the hyperparameters to reduce the AUC ROC value."
            ],
            "answer": ["B. Address data leakage by applying nested cross‐validation during model training."]
        },
        {
            "question": "You are training a ResNet model on Vertex AI using TPUs to visually categorize types of defects in automobile engines. You capture the training profile using the Cloud TPU profiler plugin and observe that it is highly input bound. You want to reduce the bottleneck and speed up your model training process. Which modifications should you make to the tf.data dataset? (Choose two.)",
            "options": [
            "A. Use the interleave option to read data.",
            "B. Set the prefetch option equal to the training batch size.",
            "C. Reduce the repeat parameters.",
            "D. Decrease the batch size argument in your transformation.",
            "E. Increase the buffer size for shuffle."
            ],
            "answer": ["A. Use the interleave option to read data.", "B. Set the prefetch option equal to the training batch size."]
        },
        {
            "question": ":red[[Suspected faulty answer]]You have been asked to develop an input pipeline for an ML training model that processes images from disparate sources at a low latency. You discover that your input data does not fit in memory. How should you create a dataset following Google-recommended best practices?",
            "options": [
            "A. Create a tf.data.Dataset.prefetch transformation.",
            "B. Convert the images into TFRecords, store the images in Cloud Storage, and then use the tf.data API to read the images for training.",
            "C. Convert the images to tf.Tensor objects, and then run Dataset.from_tensor_slices{).",
            "D. Convert data into TFRecords."
            ],
            "answer": ["A. Create a tf.data.Dataset.prefetch transformation."]
        },
        {
            "question": "Different cities in California have markedly different housing prices. Suppose you must create a model to predict housing prices. Which of the following sets of features or feature crosses could learn city‐specific relationships between roomsPerPerson and housing price?",
            "options": [
            "A. Two feature crosses: [binned latitude x binned roomsPerPerson] and [binned longitude x binned roomsPerPerson]",
            "B. Three separate binned features: [binned latitude], [binned longitude], [binned roomsPerPerson]",
            "C. One feature cross: [binned latitude x binned longitude x binned roomsPerPerson]",
            "D. One feature cross: [latitude x longitude x roomsPerPerson]"
            ],
            "answer": ["C. One feature cross: [binned latitude x binned longitude x binned roomsPerPerson]"]
        },
        {
            "question": "You are a data engineer for a finance company. You are responsible for building a unified analytics environment across a variety of on‐premises data marts. Your company is experiencing data quality and security challenges when integrating data across the servers, caused by the use of a wide range of disconnected tools and temporary solutions. You need a fully managed, cloud‐native data integration service that will lower the total cost of work and reduce repetitive work. Some members on your team prefer a codeless interface for building an extract, transform, load (ETL) process. Which service should you use?",
            "options": [
            "A. Cloud Data Fusion",
            "B. Dataprep",
            "C. Cloud Dataflow",
            "D. Apache Flink"
            ],
            "answer": ["A. Cloud Data Fusion"]
        },
        {
            "question": ":red[[Suspected faulty answer]] You work for a global footwear retailer and need to predict when an item will be out of stock based on historical inventory data. Customer behavior is highly dynamic since footwear demand is influenced by many different factors. You want to serve models that are trained on all available data but track your performance on specific subsets of data before pushing to production. What is the most streamlined, scalable, and reliable way to perform this validation?",
            "options": [
            "A. Use the tf.Transform to specify performance metrics for production readiness of the data.",
            "B. Use the entire dataset and treat the area under the receiver operating characteristic curve (AUC ROC) as the main metric.",
            "C. Use the last relevant week of data as a validation set to ensure that your model is performing accurately on current data.",
            "D. Use k‐fold cross‐validation as a validation strategy to ensure that your model is ready for production."
            ],
            "answer": ["A. Use the tf.Transform to specify performance metrics for production readiness of the data."]
        },
        {
            "question": "You are transforming a complete dataset before model training. Your model accuracy is 99 percent in training, but in production its accuracy is 66 percent. What is a possible way to improve the model in production?",
            "options": [
            "A. Apply transformation during model training.",
            "B. Perform data normalization.",
            "C. Remove missing values.",
            "D. Use tf.Transform for creating production pipelines for both training and serving."
            ],
            "answer": ["D. Use tf.Transform for creating production pipelines for both training and serving."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions_chap3):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
   
def chap4():
    st.title("Chapter 4: Choosing the Right ML Infrastructure")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Choose the right ML approach**. Understand the requirements to choose between pretrained models, AutoML, or custom models. Understand the readiness of the solution, the flexibility, and approach""\n- **Provision the right hardware for training**. Understand the various hardware options available for machine learning. Also understand the requirements of GPU and TPU hardware and the instance types that support the specialized hardware. Also learn about hardware differences in training and deployment""\n- **Provision the right hardware for predictions**. Learn the difference between provisioning during training time and during predictions. The requirements for predictions are usually scalability and the CPU and memory constraints, so CPUs and GPUs are used in the cloud. However, TPUs are used in the edge devices""\n- **Understand the available ML solutions**. Instead of provisioning hardware, take a serverless approach by using pretrained models and solutions that are built to solve a problem in a domain"

    questions_chap3 = [
        {
            "question": "Your company deals with real estate, and as part of a software development team, you have been asked to add a machine learning model to identify objects in photos uploaded to your website. How do you go about this?",
            "options": [
            "A. Use a custom model to get best results.",
            "B. Use AutoML to create object detection.",
            "C. Start with Vision AI, and if that does not work, use AutoML.",
            "D. Combine AutoML and a custom model to get better results."
            ],
            "answer": ["C. Start with Vision AI, and if that does not work, use AutoML."]
        },
        {
            "question": "Your company is working with legal documentation (thousands of pages) that needs to be translated to Spanish and French. You notice that the pretrained model in Google’s Translation AI is good, but there are a few hundred domain‐specific terms that are not translated in the way you want. You don't have any labeled data and you have only a few professional translators in your company. What do you do?",
            "options": [
            "A. Use Google's translate service and then have a human in the loop (HITL) to fix each translation.",
            "B. Use Google AutoML Translation to create a new translation model for your case.",
            "C. Use Google's Translation AI with a “glossary” of the terms you need.",
            "D. Not possible to translate because you don't seem to have data."
            ],
            "answer": ["C. Use Google's Translation AI with a “glossary” of the terms you need."]
        },
        {
            "question": "You are working with a thousand hours of video recordings in Spanish and need to create subtitles in English and French. You already have a small dataset with hundreds of hours of video for which subtitles have been created manually. What is your first approach?",
            "options": [
            "A. There is no “translated subtitle” service so use AutoML to create a “subtitle” job using the existing dataset and then use that model to create translated subtitles.",
            "B. There is no “translated subtitle” service, and there is no AutoML for this so you have to create a custom model using the data and run it on GPUs.",
            "C. There is no “translated subtitle” service, and there is no AutoML for this so you have to create a custom model using the data and run it on TPUs.",
            "D. Use the pretrained Speech‐to‐Text (STT) service and then use the pretrained Google Translate service to translate the text and insert the subtitles."
            ],
            "answer": ["D. Use the pretrained Speech‐to‐Text (STT) service and then use the pretrained Google Translate service to translate the text and insert the subtitles."]
        },
        {
            "question": "You want to build a mobile app to classify the different kinds of insects. You have enough labeled data to train but you want to go to market quickly. How would you design this?",
            "options": [
            "A. Use AutoML to train a classification model, with AutoML Edge as the method. Create an Android app using ML Kit and deploy the model to the edge device.",
            "B. Use AutoML to train a classification model, with AutoML Edge as the method. Use a Coral.ai device that has edge TPU and deploy the model on that device.",
            "C. Use AutoML to train an object detection model with AutoML Edge as the method. Use a Coral.ai device that has edge TPU and deploy the model on that device.",
            "D. Use AutoML to train an image segmentation model, with AutoML Edge as the method. Create an Android app using ML Kit and deploy the model to the edge device."
            ],
            "answer": ["A. Use AutoML to train a classification model, with AutoML Edge as the method. Create an Android app using ML Kit and deploy the model to the edge device."]
        },
        {
            "question": "You are training a deep learning model for object detection. It is taking too long to converge, so you are trying to speed up the training. While you are trying to launch an instance (with GPU) with Deep Learning VM Image, you get an error that the “NVIDIA_TESLA_V100 was not found.” What could be the problem?",
            "options": [
            "A. GPU was not available in the selected region.",
            "B. GPU quota was not sufficient.",
            "C. Preemptible GPU quota was not sufficient.",
            "D. GPU did not have enough memory."
            ],
            "answer": ["A. GPU was not available in the selected region."]
        },
        {
            "question": "Your team is building a convolutional neural network for an image segmentation problem on‐prem on a CPU‐only machine. It takes a long time to train, so you want to speed up the process by moving to the cloud. You experiment with VMs on Google Cloud to use better hardware. You do not have any code for manual placements and have not used any custom transforms. What hardware should you use?",
            "options": [
            "A. A deep learning VM with n1-standard-2 machine with 1 GPU",
            "B. A deep learning VM with more powerful e2-highCPU-16 machines",
            "C. A VM with 8 GPUs",
            "D. A VM with 1 TPU"
            ],
            "answer": ["D. A VM with 1 TPU"]
        },
        {
            "question": "You work for a hardware retail store and have a website where you get thousands of users on a daily basis. You want to display recommendations on the home page for your users, using Recommendations AI. What model would you choose?",
            "options": [
            "A. “Others you may like”",
            "B. “Frequently bought together”",
            "C. “Similar items”",
            "D. “Recommended for you”"
            ],
            "answer": ["D. “Recommended for you”"]
        },
        {
            "question": "You work for a hardware retail store and have a website where you get thousands of users on a daily basis. You want to increase your revenue by showing recommendations while customers check out. What type of model in Recommendations AI would you choose?",
            "options": [
            "A. “Others you may like”",
            "B. “Frequently bought together”",
            "C. “Similar items”",
            "D. “Recommended for you”"
            ],
            "answer": ["B. “Frequently bought together”"]
        },
        {
            "question": "You work for a hardware retail store and have a website where you get thousands of users on a daily basis. You have a customer's browsing history and want to engage the customer more. What model in Recommendations AI would you choose?",
            "options": [
            "A. “Others you may like”",
            "B. “Frequently bought together”",
            "C. “Similar items”",
            "D. “Recommended for you”"
            ],
            "answer": ["A. “Others you may like”"]
        },
        {
            "question": "You work for a hardware retail store and have a website where you get thousands of users on a daily basis. You do not have browsing events data. What type of model in Recommendations AI would you choose?",
            "options": [
            "A. “Others you may like”",
            "B. “Frequently bought together”",
            "C. “Similar items”",
            "D. “Recommended for you”"
            ],
            "answer": ["C. “Similar items”"]
        },
        {
            "question": "You work for a hardware retail store and have a website where you get thousands of users on a daily basis. You want to show details to increase cart size. You are going to use Recommendations AI for this. What model and optimization do you choose?",
            "options": [
            "A. “Others you may like” with “click‐through rate” as the objective",
            "B. “Frequently bought together” with “revenue per order” as the objective",
            "C. “Similar items” with “revenue per order” as the objective",
            "D. “Recommended for you” with “revenue per order” as the objective"
            ],
            "answer": ["B. “Frequently bought together” with “revenue per order” as the objective"]
        },
        {
            "question": "You are building a custom deep learning neural network model in Keras that will summarize a large document into a 50‐word summary. You want to try different architectures and compare the metrics and performance. What should you do?",
            "options": [
            "A. Create multiple AutoML jobs and compare performance.",
            "B. Use Cloud Composer to automate multiple jobs.",
            "C. Use the pretrained Natural Language API first.",
            "D. Run multiple jobs on the AI platform and compare results."
            ],
            "answer": ["D. Run multiple jobs on the AI platform and compare results."]
        },
        {
            "question": "You are building a sentiment analysis tool that collates the sentiment of all customer calls to the call center. The management is looking for something to measure the sentiment; it does not have to be super accurate, but it needs to be quick. What do you think is the best approach for this?",
            "options": [
            "A. Use the pretrained Natural Language API to predict sentiment.",
            "B. Use Speech‐to‐Text (STT) and then pass through the pretrained Natural Language API to predict sentiment.",
            "C. Build a custom model to predict the sentiment directly from voice calls, which captures the intonation.",
            "D. Convert Speech‐to‐Text and extract sentiment using BERT algorithm."
            ],
            "answer": ["B. Use Speech‐to‐Text (STT) and then pass through the pretrained Natural Language API to predict sentiment."]
        },
        {
            "question": "You have built a very large deep learning model using some custom TensorFlow operations written in C++ for object tracking in videos. Your model has been tested on CPU and now you want to speed up training. What would you do?",
            "options": [
            "A. Use TPU‐v4 in default setting because it involves using very large matrix operations.",
            "B. Customize the TPU‐v4 size to match with the video and recompile the custom TensorFlow operations for TPU.",
            "C. Use GPU instances because TPUs do not support custom operations.",
            "D. You cannot use GPU or TPU because neither supports custom operations."
            ],
            "answer": ["C. Use GPU instances because TPUs do not support custom operations."]
        },
        {
            "question": "You want to use GPUs for training your models that need about 50 GB of memory. What hardware options do you have?",
            "options": [
            "A. n1‐standard‐64 with 8 NVIDIA_TESLA_P100",
            "B. e2‐standard‐32 with 4 NVIDIA_TESLA_P100",
            "C. n1‐standard‐32 with 3 NVIDIA_TESLA_P100",
            "D. n2d‐standard‐32 with 4 NVIDIA_TESLA_P100"
            ],
            "answer": ["A. n1‐standard‐64 with 8 NVIDIA_TESLA_P100"]
        },
        {
            "question": "You have built a deep neural network model to translate voice in real‐time cloud TPUs and now you want to push it to your end device. What is the best option?",
            "options": [
            "A. Push the model to the end device running Edge TPU.",
            "B. Models built on TPUs cannot be pushed to the edge. The model has to be recompiled before deployment to the edge.",
            "C. Push the model to any Android device.",
            "D. Use ML Kit to reduce the size of the model to push the model to any Android device."
            ],
            "answer": ["A. Push the model to the end device running Edge TPU."]
        },
        {
            "question": "You want to use cloud TPUs and are looking at all options. Which of the below are valid options? (Choose two.)",
            "options": [
            "A. A single TPU VM",
            "B. An HPC cluster of instances with TPU",
            "C. A TPU Pod or slice",
            "D. An instance with both TPU and GPU to give additional boost"
            ],
            "answer": ["A. A single TPU VM", "C. A TPU Pod or slice"]
        },
        {
            "question": "You want to train a very large deep learning TensorFlow model (more than 100 GB) on a dataset that has a matrix in which most values are zero. You do not have any custom TensorFlow operations and have optimized the training loop to not have an I/O operation. What are your options?",
            "options": [
            "A. Use a TPU because you do not have any custom TensorFlow operations.",
            "B. Use a TPU Pod because the size of the model is very large.",
            "C. Use a GPU.",
            "D. Use an appropriately sized TPUv4 slice."
            ],
            "answer": ["C. Use a GPU."]
        },
        {
            "question": "You have been tasked to use machine learning to precisely predict the amount of liquid (down to the milliliter) in a large tank based on pictures of the tank. You have decided to use a large deep learning TensorFlow model. The model is more than 100 GB and trained on a dataset that is very large. You do not have any custom TensorFlow operations and have optimized the training loop to not have I/O operations. What are your options?",
            "options": [
            "A. Use a TPU because you do not have any custom TensorFlow operations.",
            "B. Use a TPU Pod because the size of the model is very large.",
            "C. Use a GPU.",
            "D. Use TPU‐v4 of appropriate size and shape for the use case."
            ],
            "answer": ["C. Use a GPU."]
        },
        {
            "question": "You are a data scientist trying to build a model to estimate the energy usage of houses based on photos, year built, and so on. You have built a custom model and deployed this custom container in Vertex AI. Your application is a big hit with home buyers who are using it to predict energy costs for houses before buying. You are now getting complaints that the latency is too high. To fix the latency problem, you deploy the model on a bigger instance (32‐core) but the latency is still high. What is your next step? (Choose two.)",
            "options": [
            "A. Increase the size of the instance.",
            "B. Use a GPU instance for prediction.",
            "C. Deploy the model on a computer engine instance and test the memory and CPU usage.",
            "D. Check the code to see if this is single‐threaded and other software configurations for any bugs."
            ],
            "answer": ["C. Deploy the model on a computer engine instance and test the memory and CPU usage.", "D. Check the code to see if this is single‐threaded and other software configurations for any bugs."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions_chap3):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap5():
    st.title("Chapter 5: Architecting ML Solutions")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Design reliable, scalable, and highly available ML solutions**. Understand why you need to design a scalable solution and how Google Cloud AI/ML services can help architect a scalable and highly available ML solution.""\n- **Choose an appropriate ML service**. Understand the AI/ML stack of GCP and when to use each layer of the stack based on your use case and expertise with ML.""\n- **Understand data collection and management**. Understand various types of data stores for storing your data for various ML use cases.""\n- **Know how to implement automation and orchestration**. Know when to use Vertex AI Pipelines vs. Kubeflow vs. TFX pipelines. We will cover the details in Chapter 11, “Designing ML Training”.""\n- **Understand how to best serve data**. You need to understand the best practices when deploying models. Know when to use batch prediction versus real‐time prediction and how to manage latency with online real‐time prediction."

    questions = [
        {
            "question": "You work for an online travel agency that also sells advertising placements on its website to other companies. You have been asked to predict the most relevant web banner that a user should see next. Security is important to your company. The model latency requirements are 300ms@p99, the inventory is thousands of web banners, and your exploratory analysis has shown that navigation context is a good predictor. You want to implement the simplest solution. How should you configure the prediction pipeline?",
            "options": [
            "A. Embed the client on the website, and then deploy the model on the Vertex AI platform prediction.",
            "B. Embed the client on the website, deploy the gateway on App Engine, and then deploy the model on the Vertex AI platform prediction.",
            "C. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud Bigtable for writing and for reading the user's navigation context, and then deploy the model on the Vertex AI Prediction.",
            "D. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Memorystore for writing and for reading the user's navigation context, and then deploy the model on Google Kubernetes Engine (GKE)."
            ],
            "answer": ["B. Embed the client on the website, deploy the gateway on App Engine, and then deploy the model on the Vertex AI platform prediction."]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are training a TensorFlow model on a structured dataset with 100 billion records stored in several CSV files. You need to improve the input/output execution performance. What should you do?",
            "options": [
            "A. Load the data into BigQuery and read the data from BigQuery.",
            "B. Load the data into Cloud Bigtable, and read the data from Bigtable.",
            "C. Convert the CSV files into shards of TFRecords, and store the data in Google Cloud Storage.",
            "D. Convert the CSV files into shards of TFRecords, and store the data in the Hadoop Distributed File System (HDFS)."
            ],
            "answer": ["B. Load the data into Cloud Bigtable, and read the data from Bigtable."]
        },
        {
            "question": "You are a data engineer who is building an ML model for a product recommendation system in an e‐commerce site that's based on information about logged‐in users. You will use Pub/Sub to handle incoming requests. You want to store the results for analytics and visualizing. How should you configure the pipeline?\nPub/Sub ‐> Preprocess(1) ‐> ML training/serving(2) ‐> Storage(3) ‐> Data studio/Looker studio for visualization",
            "options": [
            "A. 1 = Dataflow, 2 = Vertex Al platform, 3 = Cloud BigQuery",
            "B. 1 = Dataproc, 2 = AutoML, 3 = Cloud Memorystore",
            "C. 1 = BigQuery, 2 = AutoML, 3 = Cloud Functions",
            "D. 1 = BigQuery, 2 = Vertex Al platform, 3 = Google Cloud Storage"
            ],
            "answer": ["A. 1 = Dataflow, 2 = Vertex Al platform, 3 = Cloud BigQuery"]
        },
        {
            "question": "You are developing models to classify customer support emails. You created models with TensorFlow Estimator using small datasets on your on‐premises system, but you now need to train the models using large datasets to ensure high performance. You will port your models to Google Cloud and want to minimize code refactoring and infrastructure overhead for easier migration from on‐prem to cloud. What should you do?",
            "options": [
            "A. Use the Vertex AI platform for distributed training.",
            "B. Create a cluster on Dataproc for training.",
            "C. Create a managed instance group with autoscaling.",
            "D. Use Kubeflow Pipelines to train on a Google Kubernetes Engine cluster."
            ],
            "answer": ["A. Use the Vertex AI platform for distributed training."]
        },
        {
            "question": "You are a CTO wanting to implement a scalable solution on Google Cloud to digitize documents such as PDF files and Word DOC files in various silos. You are also looking for storage recommendations for storing the documents in a data lake. Which options have the least infrastructure efforts? (Choose two.)",
            "options": [
            "A. Use the Document AI solution.",
            "B. Use Vision AI OCR to digitize the documents.",
            "C. Use Google Cloud Storage to store documents.",
            "D. Use Cloud Bigtable to store documents.",
            "E. Use a custom Vertex AI model to build a document processing pipeline."
            ],
            "answer": ["A. Use the Document AI solution.", "C. Use Google Cloud Storage to store documents."]
        },
        {
            "question": "You work for a public transportation company and need to build a model to estimate delay for multiple transportation routes. Predictions are served directly to users in an app in real time. Because different seasons and population increases impact the data relevance, you will retrain the model every month. You want to follow Google-recommended best practices. How should you configure the end‐to‐end architecture of the predictive model?",
            "options": [
            "A. Configure Kubeflow Pipelines to schedule your multistep workflow from training to deploying your model.",
            "B. Use a model trained and deployed on BigQuery ML and trigger retraining with the scheduled query feature in BigQuery.",
            "C. Write a Cloud Functions script that launches a training and deploying job on the Vertex AI platform that is triggered by Cloud Scheduler.",
            "D. Use Cloud Composer to programmatically schedule a Dataflow job that executes the workflow from training to deploying your model."
            ],
            "answer": ["A. Configure Kubeflow Pipelines to schedule your multistep workflow from training to deploying your model."]
        },
        {
            "question": "You need to design a customized deep neural network in Keras that will predict customer purchases based on their purchase history. You want to explore model performance using multiple model architectures, store training data, and be able to compare the evaluation metrics in the same dashboard. What should you do?",
            "options": [
            "A. Create multiple models using AutoML Tables.",
            "B. Automate multiple training runs using Cloud Composer.",
            "C. Run multiple training jobs on the Vertex AI platform with similar job names.",
            "D. Create an experiment in Kubeflow Pipelines to organize multiple runs."
            ],
            "answer": ["D. Create an experiment in Kubeflow Pipelines to organize multiple runs."]
        },
        {
            "question": "You work with a data engineering team that has developed a pipeline to clean your dataset and save it in a Google Cloud Storage bucket. You have created an ML model and want to use the data to refresh your model as soon as new data is available. As part of your CI/CD workflow, you want to automatically run a Kubeflow Pipelines training job on Google Kubernetes Engine (GKE). How should you architect this workflow?",
            "options": [
            "A. Configure your pipeline with Dataflow, which saves the files in Google Cloud Storage. After the file is saved, start the training job on a GKE cluster.",
            "B. Use App Engine to create a lightweight Python client that continuously polls Google Cloud Storage for new files. As soon as a file arrives, initiate the training job.",
            "C. Configure a Google Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub–triggered Cloud Function to start the training job on a GKE cluster.",
            "D. Use Cloud Scheduler to schedule jobs at regular intervals. For the first step of the job, check the time stamp of objects in your Google Cloud Storage bucket. If there are no new files since the last run, abort the job."
            ],
            "answer": ["C. Configure a Google Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub–triggered Cloud Function to start the training job on a GKE cluster."]
        },
        {
            "question": "Your data science team needs to rapidly experiment with various features, model architectures, and hyperparameters. They need to track the accuracy metrics for various experiments and use an API to query the metrics over time. What should they use to track and report their experiments while minimizing manual effort?",
            "options": [
            "A. Use Kubeflow Pipelines to execute the experiments. Export the metrics file, and query the results using the Kubeflow Pipelines API.",
            "B. Use Vertex AI Platform Training to execute the experiments. Write the accuracy metrics to BigQuery, and query the results using the BigQuery API.",
            "C. Use Vertex AI Platform Training to execute the experiments. Write the accuracy metrics to Cloud Monitoring, and query the results using the Monitoring API.",
            "D. Use Vertex AI Workbench Notebooks to execute the experiments. Collect the results in a shared Google Sheets file, and query the results using the Google Sheets API."
            ],
            "answer": ["A. Use Kubeflow Pipelines to execute the experiments. Export the metrics file, and query the results using the Kubeflow Pipelines API."]
        },
        {
            "question": "As the lead ML Engineer for your company, you are responsible for building ML models to digitize scanned customer forms. You have developed a TensorFlow model that converts the scanned images into text and stores them in Google Cloud Storage. You need to use your ML model on the aggregated data collected at the end of each day with minimal manual intervention. What should you do?",
            "options": [
            "A. Use the batch prediction functionality of the Vertex AI platform.",
            "B. Create a serving pipeline in Compute Engine for prediction.",
            "C. Use Cloud Functions for prediction each time a new data point is ingested.",
            "D. Deploy the model on the Vertex AI platform and create a version of it for online inference."
            ],
            "answer": ["A. Use the batch prediction functionality of the Vertex AI platform."]
        },
        {
            "question": "As the lead ML architect, you are using TensorFlow and Keras as the machine learning framework and your data is stored in disk files as block storage. You are migrating to Google Cloud and you need to store the data in BigQuery as tabular storage. Which of the following techniques will you use to store TensorFlow storage data from block storage to BigQuery?",
            "options": [
            "A. tf.data.dataset reader for BigQuery",
            "B. BigQuery Python Client library",
            "C. BigQuery I/O Connector",
            "D. tf.data.iterator"
            ],
            "answer": ["B. BigQuery Python Client library"]
        },
        {
            "question": "As the CTO of the financial company focusing on building AI models for structured datasets, you decide to store most of the data used for ML models in BigQuery. Your team is currently working on TensorFlow and other frameworks. How would they modify code to access BigQuery data to build their models? (Choose three.)",
            "options": [
            "A. tf.data.dataset reader for BigQuery",
            "B. BigQuery Python Client library",
            "C. BigQuery I/O Connector",
            "D. BigQuery Omni" 
            ],
            "answer": ["A. tf.data.dataset reader for BigQuery", "B. BigQuery Python Client library", "C. BigQuery I/O Connector"]
        },
        {
            "question": "As the chief data scientist of a retail website, you develop many ML models in PyTorch and TensorFlow for Vertex AI Training. You also use Bigtable and Google Cloud Storage. In most cases, the same data is used for multiple models and projects and also updated. What is the best way to organize the data in Vertex AI?",
            "options": [
            "A. Vertex AI–managed datasets",
            "B. BigQuery",
            "C. Vertex AI Feature Store",
            "D. CSV"
            ],
            "answer": ["A. Vertex AI–managed datasets"]
        },
        {
            "question": "You are the data scientist team lead and your team is working for a large consulting firm. You are working on an NLP model to classify customer support requests. You are working on data storage strategy to store the data for NLP models. What type of storage should you avoid in a managed GCP environment in Vertex AI? (Choose two.)",
            "options": [
            "A. Block storage",
            "B. File storage",
            "C. BigQuery",
            "D. Google Cloud Storage"
            ],
            "answer": ["A. Block storage", "B. File storage"]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap6():
    st.title("Chapter 6: Building Secure ML Pipelines")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Build secure ML systems**. Understand encryption at rest and encryption in transit for Google Cloud. Know how encryption at rest and in transit works for storing data for machine learning in Cloud Storage and BigQuery. Know how you can set up IAM roles to manage your Vertex AI Workbench and how to set up network security for your Vertex AI Workbench. Last, understand some concepts such as differential privacy, federated learning, and tokenization.""\n- **Understand the privacy implications of data usage and collection**. Understand the Google Cloud Data Loss Prevention (DLP) API and how it helps identify and mask PII type data. Also, understand the Google Cloud Healthcare API to identify and mask PHI type data. Finally, understand some of the best practices for removing sensitive data."

    questions = [
        {
            "question": "You are an ML security expert at a bank that has a mobile application. You have been asked to build an ML‐based fingerprint authentication system for the app that verifies a customer's identity based on their fingerprint. Fingerprints cannot be downloaded into and stored in the bank databases. Which learning strategy should you recommend to train and deploy this ML model and make sure the fingerprints are secure and protected?",
            "options": [
            "A. Differential privacy",
            "B. Federated learning",
            "C. Tokenization",
            "D. Data Loss Prevention API"
            ],
            "answer": ["B. Federated learning"]
        },
        {
            "question": ":red[[Suspected faulty answer]] You work on a growing team of more than 50 data scientists who all use Vertex AI Workbench. You are designing a strategy to organize your jobs, models, and versions in a clean and scalable way. Which strategy is the most managed and requires the least effort?",
            "options": [
            "A. Set up restrictive IAM permissions on the Vertex AI platform notebooks so that only a single user or group can access a given instance.",
            "B. Separate each data scientist's work into a different project to ensure that the jobs, models, and versions created by each data scientist are accessible only to that user.",
            "C. Use labels to organize resources into descriptive categories. Apply a label to each created resource so that users can filter the results by label when viewing or monitoring the resources.",
            "D. Set up a BigQuery sink for Cloud Logging logs that is appropriately filtered to capture information about AI Platform resource usage. In BigQuery, create a SQL view that maps users to the resources they are using."
            ],
            "answer": ["B. Separate each data scientist's work into a different project to ensure that the jobs, models, and versions created by each data scientist are accessible only to that user."]
        },
        {
            "question": "You are an ML engineer of a Fintech company working on a project to create a model for document classification. You have a big dataset with a lot of PII that cannot be distributed or disclosed. You are asked to replace the sensitive data with specific surrogate characters. Which of the following techniques is best to use?",
            "options": [
            "A. Format‐preserving encryption or tokenization",
            "B. K‐anonymity",
            "C. Replacement",
            "D. Masking"
            ],
            "answer": ["D. Masking"]
        },
        {
            "question": "You are a data scientist of an EdTech company, and your team needs to build a model on the Vertex AI platform. You need to set up access to a Vertex AI Python library on Google Colab Jupyter Notebook. What choices do you have? (Choose three.)",
            "options": [
            "A. Create a service account key.",
            "B. Set the environment variable named GOOGLE_APPLICATION_CREDENTIALS.",
            "C. Give your service account the Vertex AI user role.",
            "D. Use console keys.",
            "E. Create a private account key."
            ],
            "answer": [
            "A. Create a service account key.",
            "B. Set the environment variable named GOOGLE_APPLICATION_CREDENTIALS.",
            "C. Give your service account the Vertex AI user role."
            ]
        },
        {
            "question": "You are a data scientist training a deep neural network. The data you are training contains PII. You have two challenges: first you need to transform the data to hide PII, and you also need to manage who has access to this data in various groups in the GCP environment. What are the choices provided by Google that you can use? (Choose two.)",
            "options": [
            "A. Network firewall",
            "B. Cloud DLP",
            "C. VPC security control",
            "D. Service keys",
            "E. Differential privacy"
            ],
            "answer": ["B. Cloud DLP", "C. VPC security control"]
        },
        {
            "question": "You are a data science manager and recently your company moved to GCP. You have to set up a JupyterLab environment for 20 data scientists on your team. You are looking for a least-managed and cost‐effective way to manage the Vertex AI Workbench so that your instances are only running when the data scientists are using the notebook. How would you architect this on GCP?",
            "options": [
            "A. Use Vertex AI–managed notebooks.",
            "B. Use Vertex AI user‐managed notebooks.",
            "C. Use Vertex AI user‐managed notebooks with a script to stop the instances when not in use.",
            "D. Use a Vertex AI pipeline."
            ],
            "answer": ["A. Use Vertex AI–managed notebooks."]
        },
        {
            "question": "You have Fast Healthcare Interoperability Resources (FHIR) data and you are building a text classification model to detect patient notes. You need to remove the PHI from the data. Which service you would use?",
            "options": [
            "A. Cloud DLP",
            "B. Cloud Healthcare API",
            "C. Cloud NLP API",
            "D. Cloud Vision AI"
            ],
            "answer": ["B. Cloud Healthcare API"]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are an ML engineer of a Fintech company building a real-time prediction engine that streams files that may contain personally identifiable information (PII) to GCP. You want to use the Cloud Data Loss Prevention (DLP) API to scan the files. How should you ensure that the PII is not accessible by unauthorized individuals?",
            "options": [
            "A. Stream all files to Google Cloud, and then write the data to BigQuery. Periodically conduct a bulk scan of the table using the DLP API.",
            "B. Stream all files to Google Cloud, and write batches of the data to BigQuery. While the data is being written to BigQuery, conduct a bulk scan of the data using the DLP API.",
            "C. Create two buckets of data: sensitive and nonsensitive. Write all data to the Nonsensitive bucket. Periodically conduct a bulk scan of that bucket using the DLP API, and move the sensitive data to the Sensitive bucket.",
            "D. Periodically conduct a bulk scan of the Google Cloud Storage bucket using the DLP API, and move the data to either the Sensitive or Nonsensitive bucket."
            ],
            "answer": ["A. Stream all files to Google Cloud, and then write the data to BigQuery. Periodically conduct a bulk scan of the table using the DLP API."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap7():
    st.title("Chapter 7: Model Building")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Choose either framework or model parallelism**. Understand multinode training strategies to train a large neural network model. The strategy can be data parallel or model parallel. Also, know what strategies can be used for distributed training of TensorFlow models""\n- **Understand modeling techniques**. Understand when to use which loss function (sparse cross‐entropy versus categorical cross‐entropy). Understand important concepts such as gradient descent, learning rate, batch size, and epoch. Also understand that these are hyperparameters and know some strategies to tune these hyperparameters to minimize loss or error rate while training your model.""\n- **Understand transfer learning**. Understand what transfer learning is and how it can help with training neural networks with limited data as these are pretrained models trained on large datasets.""\n- **Use semi‐supervised learning (SSL)**. Understand semi-supervised learning and when you need to use this method. Also know the limitations of SSL.""\n- **Use data augmentation**. You need to understand data augmentation and how you can apply it in your ML pipeline (online versus offline). You also need to learn some key data augmentation techniques such as flipping, rotation, GANs, and transfer learning.""\n- **Understand model generalization and strategies to handle overfitting and underfitting**. You need to understand bias variance trade‐off while training a neural network. Know the strategies to handle underfitting as well as strategies to handle overfitting, such as regularization. You need to understand the difference between L1 and L2 regularization and when to apply which approach."
 
    questions = [ 
        {
            "question": "Your data science team trained and tested a deep neural net regression model with good results in development. In production, six months after deployment, the model is performing poorly due to a change in the distribution of the input data. How should you address the input differences in production?",
            "options": [
            "A. Perform feature selection on the model using L1 regularization and retrain the model with fewer features.",
            "B. Retrain the model, and select an L2 regularization parameter with a hyperparameter tuning service.",
            "C. Create alerts to monitor for skew, and retrain the model.",
            "D. Retrain the model on a monthly basis with fewer features."
            ],
            "answer": ["C. Create alerts to monitor for skew, and retrain the model."]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are an ML engineer of a start‐up and have trained a deep neural network model on Google Cloud. The model has low loss on the training data but is performing worse on the validation data. You want the model to be resilient to overfitting. Which strategy should you use when retraining the model?",
            "options": [
            "A. Optimize for the **L1** regularization and dropout parameters.",
            "B. Apply an L2 regularization parameter of 0.4, and decrease the learning rate by a factor of 10.",
            "C. Apply a dropout parameter of 0.2.",
            "D. Optimize for the learning rate, and increase the number of neurons by a factor of 2."
            ],
            "answer": ["C. Apply a dropout parameter of 0.2."]
        },
        {
            "question": "You are a data scientist of a Fintech company training a computer vision model that predicts the type of government ID present in a given image using a GPU‐powered virtual machine on Compute Engine. You use the following parameters: Optimizer: SGD, Image shape = 224x224, Batch size = 64, Epochs = 10, and Verbose = 2. During training you encounter the following error: “ResourceExhaustedError: out of Memory (oom) when allocating tensor.” What should you do?",
            "options": [
            "A. Change the optimizer.",
            "B. Reduce the batch size.",
            "C. Change the learning rate.",
            "D. Reduce the image shape."
            ],
            "answer": ["B. Reduce the batch size."]
        },
        {
            "question": "You are a data science manager of an EdTech company and your team needs to build a model that predicts whether images contain a driver's license, passport, or credit card. The data engineering team already built the pipeline and generated a dataset composed of 20,000 images with driver's licenses, 2,000 images with passports, and 2,000 images with credit cards. You now have to train a model with the following label map: ['drivers_license', 'passport', 'credit_card']. Which loss function should you use?",
            "options": [
            "A. Categorical hinge",
            "B. Binary cross‐entropy",
            "C. Categorical cross‐entropy",
            "D. Sparse categorical cross‐entropy"
            ],
            "answer": ["D. Sparse categorical cross‐entropy"]
        },
        {
            "question": "You are a data scientist training a deep neural network. During batch training of the neural network, you notice that there is an oscillation in the loss. How should you adjust your model to ensure that it converges?",
            "options": [
            "A. Increase the size of the training batch.",
            "B. Decrease the size of the training batch.",
            "C. Increase the learning rate hyperparameter.",
            "D. Decrease the learning rate hyperparameter."
            ],
            "answer": ["D. Decrease the learning rate hyperparameter."]
        },
        {
            "question": "You have deployed multiple versions of an image classification model on the Vertex AI platform. You want to monitor the performance of the model versions over time. How should you perform this comparison?",
            "options": [
            "A. Compare the loss performance for each model on a held‐out dataset.",
            "B. Compare the loss performance for each model on the validation data.",
            "C. Compare the mean average precision across the models using the Continuous Evaluation feature.",
            "D. Compare the ROC curve for each model."
            ],
            "answer": ["B. Compare the loss performance for each model on the validation data."]
        },
        {
            "question": "You are training an LSTM‐based model to summarize text using the following hyperparameters: epoch = 20, batch size =32, and learning rate = 0.001. You want to ensure that training time is minimized without significantly compromising the accuracy of your model. What should you do?",
            "options": [
            "A. Modify the epochs parameter.",
            "B. Modify the batch size parameter.",
            "C. Modify the learning rate parameter.",
            "D. Increase the number of epochs."
            ],
            "answer": ["B. Modify the batch size parameter."]
        },
        {
            "question": "Your team needs to build a model that predicts whether images contain a driver's license or passport. The data engineering team already built the pipeline and generated a dataset composed of 20,000 images with driver's licenses and 5,000 images with passports.  You have transformed the features into one‐hot encoded value for training. You now have to train a model to classify these two classes; which loss function should you use?",
            "options": [
            "A. Sparse categorical cross‐entropy",
            "B. Categorical cross‐entropy",
            "C. Categorical hinge",
            "D. Binary cross‐entropy"
            ],
            "answer": ["B. Categorical cross‐entropy"]
        },
        {
            "question": "You have developed your own DNN model with TensorFlow to identify products for an industry. During training, your custom model converges but the tests are giving unsatisfactory results. What do you think is the problem and how can you fix it? (Choose two.)",
            "options": [
            "A. You have to change the algorithm to XGBoost.",
            "B. You have an overfitting problem.",
            "C. You need to increase your learning rate hyperparameter.",
            "D. The model is complex and you need to regularize the model using L2.",
            "E. Reduce the batch size."
            ],
            "answer": ["B. You have an overfitting problem.", "D. The model is complex and you need to regularize the model using L2."]
        },
        {
            "question": "As the lead ML engineer for your company, you are building a deep neural network TensorFlow model to optimize customer satisfaction. Your focus is to minimize bias and increase accuracy for the model. Which other parameter do you need to consider so that your model converges while training and doesn't lead to underfit or overfit problems?",
            "options": [
            "A. Learning rate",
            "B. Batch size",
            "C. Variance",
            "D. Bagging"
            ],
            "answer": ["C. Variance"]
        },
        {
            "question": "As a data scientist, you are working on building a DNN model for text classification using Keras TensorFlow. Which of the following techniques should *not* be used? (Choose two.)",
            "options": [
            "A. Softmax function",
            "B. Categorical cross‐entropy",
            "C. Dropout layer",
            "D. L1 regularization",
            "E. K‐means"
            ],
            "answer": ["D. L1 regularization", "E. K‐means"]
        },
        {
            "question": "As the ML developer for a gaming company, you are asked to create a game in which the characters look like human players. You have been asked to generate the avatars for the game. However, you have very limited data. Which technique would you use?",
            "options": [
            "A. Feedforward neural network",
            "B. Data augmentation",
            "C. Recurrent neural network",
            "D. Transformers"
            ],
            "answer": ["B. Data augmentation"]
        },
        {
            "question": "You are working on building a TensorFlow model for binary classification with a lot of categorical features. You have to encode them with a limited set of numbers. Which *activation function* will you use for the task?",
            "options": [
            "A. One‐hot encoding",
            "B. Sigmoid",
            "C. Embeddings",
            "D. Feature cross"
            ],
            "answer": ["B. Sigmoid"]
        },
        {
            "question": "You are the data scientist working on building a TensorFlow model to optimize the level of customer satisfaction for after-sales service. You are struggling with learning rate, batch size, and epoch to optimize and converge your model. What is your problem in ML?",
            "options": [
            "A. Regularization",
            "B. Hyperparameter tuning",
            "C. Transformer",
            "D. Semi-supervised learning"
            ],
            "answer": ["B. Hyperparameter tuning"]
        },
        {
            "question": "You are a data scientist working for a start‐up on several projects with TensorFlow. You need to increase the performance of the training and you are already using caching and prefetching. You want to use GPU for training but you have to use only one machine to be cost‐effective. Which of the following tf distribution strategies should you use?",
            "options": [
            "A. MirroredStrategy",
            "B. MultiWorkerMirroredStrategy",
            "C. TPUStrategy",
            "D. ParameterServerStrategy"
            ],
            "answer": ["A. MirroredStrategy"]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap8():
    st.title("Chapter 8: Model Training and Hyperparameter Tuning")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Know how to ingest various file types into training**. Understand the various file types, such as structured (for example, CSV), unstructured (for example, text files), and semi-structured (for example, JSON files). Know how these file types can be stored and ingested for AI/ML workloads in GCP. Understand how the file ingestion into Google Cloud works by using a Google Cloud data analytics platform into stages such as collect, process, store, and analyze. For collecting data into Google Cloud Storage, you can use Pub/Sub and Pub/Sub Lite to collect real-time data as well as BigQuery Data Transfer Service and Datastream to migrate data from third-party sources and databases to Google Cloud. In the process phase, understand how we can transform the data or run Spark/Hadoop jobs for ETL using services such as Cloud Dataflow, Cloud Data Fusion, Cloud Dataproc, Cloud Composer, and Cloud Dataprep.""\n- **Know how to use the Vertex AI Workbench environment by using common frameworks**. Understand the feature differences and framework supported by both managed and user-managed notebooks. Understand when you should use user-managed notebooks versus managed notebooks. Understand how to create these notebooks and what features they support out of the box.""\n- **Know how to train a model as a job in different environments**. Understand options for Vertex AI training such as AutoML and custom training. Then understand how you can perform custom training by using either a prebuilt container or a custom container using Vertex AI training along with architecture. Understand using a training pipeline versus custom jobs to set up training in Vertex AI. Vertex AI training supports frameworks such as scikit-learn, TensorFlow, PyTorch, and XGBoost. Also, understand how to set up distributed training using Vertex AI custom jobs.""\n- **Be able to unit test for model training and serving**. Understand why and how you can unit test the data and model for machine learning. Understand how to test for updates in APIs after model endpoints are updated and how to test for algorithm correctness..""\n- **Understand hyperparameter tuning**. Understand hyperparameter tuning and various search algorithms for hyperparameter tuning such as grid search, random search, and Bayesian search. Understand when to use which search algorithm to speed up performance. Know how to set up hyperparameter tuning using custom jobs. Last, also understand Vertex AI Vizier and how it's different from setting up hyperparameter tuning.""\n- **Track metrics during training**. You can use Interactive shell, Tensorflow Profiler and What-If tool to track metrics during model training.""\n- **Conduct a retraining/redeployment evaluation**. Understand bias variance trade-off while training a neural network. Then you need to understand strategies to handle underfitting and strategies to handle overfitting, such as regularization. Know the difference between L1 and L2 regularization and when to apply which approach."
 
    questions = [ 
        {
            "question": "You are a data scientist for a financial firm who is developing a model to classify customer support emails. You created models with TensorFlow Estimators using small datasets on your on‐premises system, but you now need to train the models using large datasets to ensure high performance. You will port your models to Google Cloud and want to minimize code refactoring and infrastructure overhead for easier migration from on‐prem to cloud. What should you do?",
            "options": [
            "A. Use Vertex AI custom jobs for training.",
            "B. Create a cluster on Dataproc for training.",
            "C. Create an AutoML model using Vertex AI training.",
            "D. Create an instance group with autoscaling."
            ],
            "answer": ["A. Use Vertex AI custom jobs for training."]
        },
        {
            "question": "You are a data engineer building a demand‐forecasting pipeline in production that uses Dataflow to preprocess raw data prior to model training and prediction. During preprocessing, you perform z‐score normalization on data stored in BigQuery and write it back to BigQuery. Because new training data is added every week, what should you do to make the process more efficient by minimizing computation time and manual intervention?",
            "options": [
            "A. Translate the normalization algorithm into SQL for use with BigQuery.",
            "B. Normalize the data with Apache Spark using the Dataproc connector for BigQuery.",
            "C. Normalize the data with TensorFlow data transform.",
            "D. Normalize the data by running jobs in Google Kubernetes Engine clusters."
            ],
            "answer": ["A. Translate the normalization algorithm into SQL for use with BigQuery."]
        },
        {
            "question": "You are an ML engineer for a fashion apparel company designing a customized deep neural network in Keras that predicts customer purchases based on their purchase history. You want to explore model performance using multiple model architectures, to store training data, and to compare the evaluation metric *while the job is running*. What should you do?",
            "options": [
            "A. Create multiple models using AutoML Tables.",
            "B. Create an experiment in Kubeflow Pipelines to organize multiple runs.",
            "C. Run multiple training jobs on the Vertex AI platform with an interactive shell enabled.",
            "D. Run multiple training jobs on the Vertex AI platform with hyperparameter tuning."
            ],
            "answer": ["C. Run multiple training jobs on the Vertex AI platform with an interactive shell enabled."]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are a data scientist who has created an ML pipeline with hyperparameter tuning jobs using Vertex AI custom jobs. One of your tuning jobs is taking longer than expected and delaying the downstream processes. You want to speed up the tuning job without significantly compromising its effectiveness. Which actions should you take? (Choose three.)",
            "options": [
            "A. Decrease the number of parallel trials.",
            "B. Change the search algorithm from grid search to random search.",
            "C. Decrease the range of floating‐point values.",
            "D. Change the algorithm to grid search.",
            "E. Set the early stopping parameter to TRUE."
            ],
            "answer": [
            "A. Decrease the number of parallel trials.",
            "B. Change the search algorithm from grid search to random search.",
            "C. Decrease the range of floating‐point values."
            ]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are a data engineer using PySpark data pipelines to conduct data transformations at scale on Google Cloud. However, your pipelines are taking over 12 hours to run. In order to expedite pipeline runtime, you do not want to manage servers and need a tool that can run SQL. You have already moved your raw data into Cloud Storage. How should you build the pipeline on Google Cloud while meeting speed and processing requirements?",
            "options": [
            "A. Use Data Fusion's GUI to build the transformation pipelines, and then write the data into BigQuery.",
            "B. Convert your PySpark commands into Spark SQL queries to transform the data and then run your pipeline on Dataproc to write the data into BigQuery using BigQuery Spark connector.",
            "C. Ingest your data into BigQuery from Cloud Storage, convert your PySpark commands into BigQuery SQL queries to transform the data, and then write the transformations to a new table.",
            "D. Ingest your data into Cloud SQL, convert your PySpark commands into Spark SQL queries to transform the data, and then use SQL queries from BigQuery for machine learning."
            ],
            "answer": ["B. Convert your PySpark commands into Spark SQL queries to transform the data and then run your pipeline on Dataproc to write the data into BigQuery using BigQuery Spark connector."]
        },
        {
            "question": "You are a lead data scientist manager who is managing a team of data scientists using a cloud‐based system to submit training jobs. This system has become very difficult to administer. The data scientists you work with use many different frameworks such as Keras, PyTorch, Scikit, and custom libraries. What is the most managed way to run the jobs in Google Cloud?",
            "options": [
            "A. Use the Vertex AI training custom containers to run training jobs using any framework.",
            "B. Use the Vertex AI training prebuilt containers to run training jobs using any framework.",
            "C. Configure Kubeflow to run on Google Kubernetes Engine and receive training jobs through TFJob.",
            "D. Create containerized images on Compute Engine using GKE and push these images on a centralized repository."
            ],
            "answer": ["A. Use the Vertex AI training custom containers to run training jobs using any framework."]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are training a TensorFlow model on a structured dataset with 500 billion records stored in several CSV files. You need to improve the input/output execution performance. What should you do?",
            "options": [
            "A. Load the data into HDFS.",
            "B. Load the data into Cloud Bigtable, and read the data from Bigtable using a TF Bigtable connector.",
            "C. Convert the CSV files into shards of TFRecords, and store the data in Cloud Storage.",
            "D. Load the data into BigQuery using Dataflow jobs."
            ],
            "answer": ["B. Load the data into Cloud Bigtable, and read the data from Bigtable using a TF Bigtable connector."]
        },
        {
            "question": "You are the senior solution architect of a gaming company. You have to design a streaming pipeline for ingesting player interaction data for a mobile game. You want to perform ML on the streaming data. What should you do to build a pipeline with the least overhead?",
            "options": [
            "A. Use Pub/Sub with Cloud Dataflow streaming pipeline to ingest data.",
            "B. Use Apache Kafka with Cloud Dataflow streaming pipeline to ingest data.",
            "C. Use Apache Kafka with Cloud Dataproc to ingest data.",
            "D. Use Pub/Sub Lite streaming connector with Cloud Data Fusion."
            ],
            "answer": ["A. Use Pub/Sub with Cloud Dataflow streaming pipeline to ingest data."]
        },
        {
            "question": "You are a data scientist working on a smart city project to build an ML model to detect anomalies in real‐time sensor data. You will use Pub/Sub to handle incoming requests. You want to store the results for analytics and visualization. How should you configure the below pipeline:\nIngest data using Pub/Sub‐> 1. Preprocess ‐> 2. ML training ‐> 3. Storage ‐> Visualization in Data Studio",
            "options": [
            "A. 1. Dataflow, 2. Vertex AI Training, 3. BigQuery",
            "B. 1. Dataflow, 2. Vertex AI AutoML, 3. Bigtable",
            "C. 1. BigQuery, 2. Vertex AI Platform, 3. Cloud Storage",
            "D. 1. Dataflow, 2. Vertex AI AutoML, 3. Cloud Storage"
            ],
            "answer": ["A. 1. Dataflow, 2. Vertex AI Training, 3. BigQuery"]
        },
        {
            "question": "You are a data scientist who works for a Fintech company. You want to understand how effective your company's latest advertising campaign for a financial product is. You have streamed 900 MB of campaign data into BigQuery. You want to query the table and then manipulate the results of that query with a pandas DataFrame in a Vertex AI platform notebook. What will be the least number of steps needed to do this?",
            "options": [
            "A. Download your table from BigQuery as a local CSV file, and upload it to your AI platform notebook instance. Use pandas read_csv to ingest the file as a pandas DataFrame.",
            "B. Export your table as a CSV file from BigQuery to Google Drive, and use the Google Drive API to ingest the file into your notebook instance.",
            "C. Use the Vertex AI platform notebook's BigQuery cell magic to query the data, and ingest the results as a pandas DataFrame using pandas BigQuery client.",
            "D. Use the bq extract command to export the table as a CSV file to Cloud Storage, and then use gsui cp to copy the data into the notebook Use pandas read_csv to ingest the file."
            ],
            "answer": ["C. Use the Vertex AI platform notebook's BigQuery cell magic to query the data, and ingest the results as a pandas DataFrame using pandas BigQuery client."]
        },
        {
            "question": "You are a data scientist working on a fraud detection model. You will use Pub/Sub to handle incoming requests. You want to store the results for analytics and visualization. How should you configure the following pipeline: 1. Ingest data ‐> 2. Preprocess ‐> 3. ML training and *visualize in Data/Looker Studio*",
            "options": [
            "A. 1. Dataflow, 2. Vertex AI Training, 3. BigQuery",
            "B. 1. Pub/Sub, 2. Dataflow, 3. BigQuery ML",
            "C. 1. Pub/Sub, 2. Dataflow, 3. Vertex AI Training",
            "D. 1. Dataflow, 2. Vertex AI AutoML, 3. Cloud Storage"
            ],
            "answer": ["B. 1. Pub/Sub, 2. Dataflow, 3. BigQuery ML"]
        },
        {
            "question": ":red[[Suspected faulty answer]]  You are an ML engineer working for a public health team to create a pipeline to classify support tickets on Google Cloud. You analyzed the requirements and decided to use TensorFlow to build the classifier so that you have full control of the model's code, serving, and deployment. You will use Kubeflow Pipelines for the ML platform. To save time, you want to build on existing resources and use managed services instead of building a completely new model. How should you build the classifier?",
            "options": [
            "A. Use an established text classification model and train using Vertex AI Training as is to classify support requests.",
            "B. Use an established text classification model and train using Vertex AI Training to perform transfer learning.",
            "C. Use AutoML Natural Language to build the support requests classifier.",
            "D. Use the Natural Language API to classify support requests."
            ],
            "answer": ["A. Use an established text classification model and train using Vertex AI Training as is to classify support requests."]
        },
        {
            "question": "You are training a TensorFlow model for binary classification with a lot of categorical features using Vertex AI custom jobs. You are looking for UI tools to track metrics of your model such as CPU utilization and network I/O and features used while training. Which tools will you pick? (Choose two.)",
            "options": [
            "A. Interactive shell",
            "B. TensorFlow Profiler",
            "C. Jupyter Notebooks",
            "D. Looker Studio",
            "E. Looker"
            ],
            "answer": ["A. Interactive shell", "B. TensorFlow Profiler"]
        },
        {
            "question": "You are training a TensorFlow model to identify semi‐finished products using Vertex AI custom jobs. You want to monitor the performance of the model. Which of the following can you use?",
            "options": [
            "A. TensorFlow Profiler",
            "B. TensorFlow Debugger",
            "C. TensorFlow Trace",
            "D. TensorFlow Checkpoint"
            ],
            "answer": ["A. TensorFlow Profiler"]
        },
        {
            "question": "You are a data scientist working for a start‐up on several projects with TensorFlow. Your data is in Parquet format and you need to manage input and output. You are looking for the most cost‐effective solution to manage the input while training TensorFlow models on Google Cloud. Which of the following should you use?",
            "options": [
            "A. TensorFlow I/O",
            "B. Cloud Dataproc",
            "C. Cloud Dataflow",
            "D. BigQuery to TFRecords"
            ],
            "answer": ["A. TensorFlow I/O"]
        },
        {
            "question": "You are training a TensorFlow model for binary classification with many categorical features using Vertex AI custom jobs. Your manager has asked you about the classification metric and also to explain the inference. You would like to show them an interactive demo with visual graphs. Which tool should you use?",
            "options": [
            "A. TensorBoard",
            "B. What‐If Tool",
            "C. Looker",
            "D. Language Interpretability Tool (LIT)"
            ],
            "answer": ["B. What‐If Tool"]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap9():
    st.title("Chapter 9: Model Explainability on Vertex AI")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "**Understand model explainability on Vertex AI**. Know what explainability is and the difference between global and local explanations. Why is it important to explain models? What is feature importance? Understand the options of feature attribution on the Vertex AI platform such as Sampled Shapley algorithm, integrated gradients, and XRAI. We covered data bias and fairness and how feature attributions can help with determining bias and fairness from the data. ML Solution readiness talks about Responsible AI and ML model governance best practices. Understand that explainable AI in Vertex AI is supported for the TensorFlow prediction container using the Explainable AI SDK and for the Vertex AI AutoML tabular and AutoML image models."
 
    questions = [
        {
            "question": ":red[[Suspected faulty answer]] You are a data scientist building a linear model with more than 100 input features, all with values between –1 and 1. You suspect that many features are non‐informative. You want to remove the noninformative features from your model while keeping the informative ones in their original form. Which technique should you use?",
            "options": [
            "A. Use principal component analysis to eliminate the least informative features.",
            "B. When building your model, use Shapley values to determine which features are the most informative.",
            "C. Use L1 regularization to reduce the coefficients of noninformative features to 0.",
            "D. Use an iterative dropout technique to identify which features do not degrade the model when removed."
            ],
            "answer": ["B. When building your model, use Shapley values to determine which features are the most informative."]
        },
        {
            "question": "You are a data scientist at a startup and your team is working on a number of ML projects. Your team trained a TensorFlow deep neural network model for image recognition that works well and is about to be rolled out in production. You have been asked by leadership to demonstrate the inner workings of the model. What explainability technique would you use on Google Cloud?",
            "options": [
            "A. Sampled Shapley",
            "B. Integrated gradient",
            "C. PCA",
            "D. What‐If Tool analysis"
            ],
            "answer": ["B. Integrated gradient"]
        },
        {
            "question": "You are a data scientist working with Vertex AI and want to leverage Explainable AI to understand which are the most essential features and how they impact model predictions. Select the model types and services supported by Vertex Explainable AI. (Choose three.)",
            "options": [
            "A. AutoML Tables",
            "B. Image classification",
            "C. Custom DNN models",
            "D. Decision trees",
            "E. Linear learner"
            ],
            "answer": ["A. AutoML Tables", "B. Image classification", "C. Custom DNN models"]
        },
        {
            "question": "You are an ML engineer working with Vertex Explainable AI. You want to understand the most important features for training models that use image and tabular datasets. Which of the feature attribution techniques can you use? (Choose three.)",
            "options": [
            "A. XRAI",
            "B. Sampled Shapley",
            "C. Minimum likelihood",
            "D. Interpretability",
            "E. Integrated gradients"
            ],
            "answer": ["A. XRAI", "B. Sampled Shapley", "E. Integrated gradients"]
        },
        {
            "question": "You are a data scientist training a TensorFlow model with graph operations as operations that perform decoding and rounding tasks. Which technique would you use to debug or explain this model in Vertex AI?",
            "options": [
            "A. Sampled Shapley",
            "B. Integrated gradients",
            "C. XRAI",
            "D. PCA"
            ],
            "answer": ["A. Sampled Shapley"]
        },
        {
            "question": "You are a data scientist working on creating an image classification model on Vertex AI. You want these images to have feature attribution. Which of the attribution techniques is supported by Vertex AI AutoML images? (Choose two.)",
            "options": [
            "A. Sampled Shapely",
            "B. Integrated gradients",
            "C. XRAI",
            "D. DNN"
            ],
            "answer": ["B. Integrated gradients", "C. XRAI"]
        },
        {
            "question": "You are a data scientist working on creating an image classification model on Vertex AI. You want to set up an explanation for testing your TensorFlow code in user‐managed notebooks. What is the suggested approach with the least effort?",
            "options": [
            "A. Set up local explanations using Explainable AI SDK in the notebooks.",
            "B. Configure explanations for the custom TensorFlow model.",
            "C. Set up an AutoML classification model to get explanations.",
            "D. Set the generateExplanation field to true when you create a batch prediction job."
            ],
            "answer": ["A. Set up local explanations using Explainable AI SDK in the notebooks."]
        },
        {
            "question": "You are a data scientist who works in the aviation industry. You have been given a task to create a model to identify planes. The images in the dataset are of poor quality. Your model is identifying birds as planes. Which approach would you use to help explain the predictions with this dataset?",
            "options": [
            "A. Use Vertex AI example–based explanations.",
            "B. Use the integrated gradients technique for explanations.",
            "C. Use the Sampled Shapley technique for explanations.",
            "D. Use the XRAI technique for explanations."
            ],
            "answer": ["A. Use Vertex AI example–based explanations."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap10():
    st.title("Chapter 10: Scaling Models in Production")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Understand TensorFlow Serving**. Understand what TensorFlow Serving is and how to deploy a trained TensorFlow model using TF Serving. Know the different ways to set up TF Serving with Docker. Understand the TF Serving prediction response based on a saved model's SignatureDef tensors""\n- **Understand the scaling prediction services (online, batch, and caching)**. Understand the difference between online batch and caching. For online serving, understand the differences in architecture and use cases with respect to input features that are fetched in real time to invoke the model for prediction (static reference features and dynamic reference features). Also, understand the caching strategies to improve serving latency""\n- **Understand the Google Cloud serving options**. Understand how to set up real‐time endpoints using Google Cloud Vertex AI Prediction for custom models or models trained outside Vertex AI; understand how to set up predictions using both APIs and the GCP console setup. Also, understand how to set up a batch job for any model using Vertex AI batch prediction""\n- **Test for target performance**. Understand why model performance in production degrades. Also understand at a high level how Vertex AI services such as Vertex AI Model Monitoring can help with performance degradation issues""\n- **Configure triggers and pipeline schedules**. Understand ways to set up a trigger to invoke a trained model or deploy a model for prediction on Google Cloud. Know how to schedule the triggers, such as using Cloud Scheduler and the Vertex AI managed notebooks scheduler. Also, learn how to automate the pipeline with Workflows, Vertex AI Pipelines, and Cloud Composer"
 
    questions = [
        {
            "question": ":red[[Suspected faulty answer]] You are a data scientist working for an online travel agency. You have been asked to predict the most relevant web banner that a user should see next in near real time. The model latency requirements are 300ms@p99, and the inventory is thousands of web banners. You want to implement the simplest solution on Google Cloud. How should you configure the prediction pipeline?",
            "options": [
            "A. Embed the client on the website, and cache the predictions in a data store by creating a batch prediction job pointing to the data warehouse. Deploy the gateway on App Engine, and then deploy the model using Vertex AI Prediction.",
            "B. Deploy the model using TF Serving.",
            "C. Deploy the model using the Google Kubernetes engine.",
            "D. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud Bigtable for writing and for reading the user's navigation context, and then deploy the model on Vertex AI."
            ],
            "answer": ["D. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud Bigtable for writing and for reading the user's navigation context, and then deploy the model on Vertex AI."]
        },
        {
            "question": "You are a data scientist training a text classification model in TensorFlow using the Vertex AI platform. You want to use the trained model for batch predictions on text data stored in BigQuery while minimizing computational overhead. What should you do?",
            "options": [
            "A. Submit a batch prediction job on Vertex AI that points to input data as a BigQuery table where text data is stored.",
            "B. Deploy and version the model on the Vertex AI platform.",
            "C. Use Dataflow with the SavedModel to read the data from BigQuery.",
            "D. Export the model to BigQuery ML."
            ],
            "answer": ["A. Submit a batch prediction job on Vertex AI that points to input data as a BigQuery table where text data is stored."]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are a CTO of a global bank and you appointed an ML engineer to build an application for the bank that will be used by millions of customers. Your team has built a forecasting model that predicts customers' account balances three days in the future. Your team will use the results in a new feature that will notify users when their account balance is likely to drop below a certain amount. How should you serve your predictions?",
            "options": [
            "A. Create a Pub/Sub topic for each user. Deploy a Cloud Function that sends a notification when your model predicts that a user's account balance will drop below the threshold.",
            "B. Create a Pub/Sub topic for each user. Deploy an application on the App Engine environment that sends a notification when your model predicts that a user's account balance will drop below the threshold.",
            "C. Build a notification system on Firebase. Register each user with a user ID on the Firebase Cloud Messaging server, which sends a notification when the average of all account balance predictions drops below the threshold.",
            "D. Build a notification system on a Docker container. Set up cloud functions and Pub/Sub, which sends a notification when the average of all account balance predictions drops below the threshold."
            ],
            "answer": ["A. Create a Pub/Sub topic for each user. Deploy a Cloud Function that sends a notification when your model predicts that a user's account balance will drop below the threshold."]
        },
        {
            "question": "You are a data scientist and you trained a text classification model using TensorFlow. You have downloaded the saved model for TF Serving. The model has the following SignatureDefs:\n\n ```\n inputs['text'] tensor_info:\n dtype: String\n shape: (-1, 2)\n name: dnn/head/predictions/textclassifier\n```\nSignatureDefs for output.:\n\n ```\noutput ['text'] tensor_info:\n dtype: String\n shape: (-1, 2)\n name: tfserving/predict\n```\n\nWhat is the correct way to write the predict request?",
            "options": [
            "A. `data = json.dumps({\"signature_name\": “seving_default”, “instances” [[‘ab’, ‘bc’, ‘cd’]]})`",
            "B. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’, ‘c’, ‘d’, ‘e’, ‘f’]]})`",
            "C. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’, ‘c’], [‘d’, ‘e’, ‘f’]]})`",
            "D. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’], [‘c’, ‘d’], [‘e’, ‘f’]]})`"
            ],
            "answer": ["D. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’], [‘c’, ‘d’], [‘e’, ‘f’]]})`"]
        },
        {
            "question": ":red[[Suspected faulty answer]] You are an ML engineer who has trained a model on a dataset that required computationally expensive preprocessing operations. You need to execute the same preprocessing at prediction time. You deployed the model on the Vertex AI platform for high‐throughput online prediction. Which architecture should you use?",
            "options": [
            "A. Send incoming prediction requests to a Pub/Sub topic. Set up a Cloud Function that is triggered when messages are published to the Pub/Sub topic. Implement your preprocessing logic in the Cloud Function. Submit a prediction request to the Vertex AI platform using the transformed data. Write the predictions to an outbound Pub/Sub queue.",
            "B. Stream incoming prediction request data into Cloud Spanner. Create a view to abstract your preprocessing logic. Query the view every second for new records. Submit a prediction request to the Vertex AI platform using the transformed data. Write the predictions to an outbound Pub/Sub queue.",
            "C. Send incoming prediction requests to a Pub/Sub topic. Transform the incoming data using a Dataflow job. Submit a prediction request to the Vertex AI platform using the transformed data. Write the predictions to an outbound Pub/Sub queue.",
            "D. Validate the accuracy of the model that you trained on preprocessed data. Create a new model that uses the raw data and is available in real time. Deploy the new model on to the Vertex AI platform for online prediction."
            ],
            "answer": ["A. Send incoming prediction requests to a Pub/Sub topic. Set up a Cloud Function that is triggered when messages are published to the Pub/Sub topic. Implement your preprocessing logic in the Cloud Function. Submit a prediction request to the Vertex AI platform using the transformed data. Write the predictions to an outbound Pub/Sub queue."]
        },
        {
            "question": "As the lead data scientist for your company, you are responsible for building ML models to digitize scanned customer forms. You have developed a TensorFlow model that converts the scanned images into text and stores them in Cloud Storage. You need to use your ML model on the aggregated data collected at the end of each day with minimal manual intervention. What should you do?",
            "options": [
            "A. Use the batch prediction functionality of the Vertex AI platform.",
            "B. Create a serving pipeline in Compute Engine for prediction.",
            "C. Use Cloud Functions for prediction each time a new data point is ingested.",
            "D. Deploy the model on the Vertex AI platform and create a version of it for online inference."
            ],
            "answer": ["A. Use the batch prediction functionality of the Vertex AI platform."]
        },
        {
            "question": "As the lead data scientist for your company, you need to create a schedule to run batch jobs using the Jupyter Notebook at the end of each day with minimal manual intervention. What should you do?",
            "options": [
            "A. Use the schedule function in Vertex AI managed notebooks.",
            "B. Create a serving pipeline in Compute Engine for prediction.",
            "C. Use Cloud Functions for prediction each time a new data point is ingested.",
            "D. Use Cloud Workflow to schedule the batch prediction Vertex AI job by cloud function."
            ],
            "answer": ["A. Use the schedule function in Vertex AI managed notebooks."]
        },
        {
            "question": "You are a data scientist working for an online travel agency. Your management has asked you to predict the most relevant news article that a user should see next in near real time. The inventory is in a data warehouse, which has thousands of news articles. You want to implement the simplest solution on Google Cloud with the least latency while serving the model. How should you configure the prediction pipeline?",
            "options": [
            "A. Embed the client on the website, deploy the gateway on App Engine, and then deploy the model using Vertex AI Prediction.",
            "B. Deploy the model using TF Serving.",
            "C. Deploy the model using Google Kubernetes Engine.",
            "D. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud Bigtable for writing and for reading the user's navigation context, and then deploy the model on Vertex"
            ],
            "answer": ["D. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud Bigtable for writing and for reading the user's navigation context, and then deploy the model on Vertex"]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap11():
    st.title("Chapter 11: Designing ML Training Pipelines")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Understand the different orchestration frameworks**. Know what an orchestration framework is and why it's needed. You should know what Kubeflow Pipelines is and how you can run Kubeflow Pipelines on GCP. You should also know Vertex AI Pipelines and how you can run Kubeflow and TFX on Vertex AI Pipelines. Also learn about Apache Airflow and Cloud Composer. Finally, compare and contrast all four orchestration methods for automating ML workflows"
    "\n- **Identify the components, parameters, triggers, and compute needs on these frameworks**. Know ways to schedule ML workflows using Kubeflow and Vertex AI Pipelines. For Kubeflow, understand how you would use Cloud Build to trigger a deployment. For Vertex AI Pipelines, understand how you can use Cloud Function event triggers to schedule the pipeline"
    "\n- **Understand the system design of TFX/Kubeflow. Know system design with Kubeflow and TensorFlow**. Understand that in Kubeflow Pipelines, you create every task into a component and orchestrate the components. Understand how you can run TFX pipelines on Kubeflow and how to use TFX components and TFX libraries to define ML pipelines. Understand that to orchestrate ML pipelines using TFX, you can use any runtime or orchestrator such as Kubeflow or Apache Airflow. You can also run TFX on GCP using Vertex AI Pipelines"
 
    questions = [
        {
            "question": "You are a data scientist building a TensorFlow model with more than 100 input features, all with values between –1 and 1. You want to serve models that are trained on all available data but track your performance on specific subsets of data before pushing to production. What is the most streamlined and reliable way to perform this validation?",
            "options": [
            "A. Use the TFX ModelValidator component to specify performance metrics for production readiness.",
            "B. Use the entire dataset and treat the area under the curve receiver operating characteristic (AUC ROC) as the main metric.",
            "C. Use L1 regularization to reduce the coefficients of uninformative features to 0.",
            "D. Use k‐fold cross‐validation as a validation strategy to ensure that your model is ready for production."
            ],
            "answer": ["A. Use the TFX ModelValidator component to specify performance metrics for production readiness."]
        },
        {
            "question": "Your team has developed an ML pipeline using Kubeflow to clean your dataset and save it in a Google Cloud Storage bucket. You created an ML model and want to use the data to refresh your model as soon as new data is available. As part of your CI/CD workflow, you want to automatically run a Kubeflow Pipelines job on GCP. How should you design this workflow with the least effort and in the most managed way?",
            "options": [
            "A. Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub–triggered Cloud Function to start the Vertex AI Pipelines.",
            "B. Use Cloud Scheduler to schedule jobs at a regular interval. For the first step of the job, check the time stamp of objects in your Cloud Storage bucket. If there are no new files since the last run, abort the job.",
            "C. Use App Engine to create a lightweight Python client that continuously polls Cloud Storage for new files. As soon as a file arrives, initiate the Kubeflow Pipelines job on GKE.",
            "D. Configure your pipeline with Dataflow, which saves the files in Cloud Storage. After the file is saved, you start the job on GKE."
            ],
            "answer": ["A. Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub–triggered Cloud Function to start the Vertex AI Pipelines."]
        },
        {
            "question": "You created an ML model and want to use the data to refresh your model as soon as new data is available in a Google Cloud Storage bucket. As part of your CI/CD workflow, you want to automatically run a Kubeflow Pipelines training job on GKE. How should you design this workflow with the least effort and in the most managed way?",
            "options": [
            "A. Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub–triggered Cloud Function to start the training job on GKE.",
            "B. Use Cloud Scheduler to schedule jobs at a regular interval. For the first step of the job, check the time stamp of objects in your Cloud Storage bucket to see if there are no new files since the last run.",
            "C. Use App Engine to create a lightweight Python client that continuously polls Cloud Storage for new files. As soon as a file arrives, initiate the Kubeflow Pipelines job on GKE.",
            "D. Configure your pipeline with Dataflow, which saves the files in Cloud Storage. After the file is saved, you can start the job on GKE."
            ],
            "answer": ["A. Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub–triggered Cloud Function to start the training job on GKE."]
        },
        {
            "question": "You are an ML engineer for a global retail company. You are developing a Kubeflow pipeline on Google Kubernetes Engine for a recommendation system. The first step in the pipeline is to issue a query against BigQuery. You plan to use the results of that query as the input to the next step in your pipeline. Choose two ways you can create this pipeline.",
            "options": [
            "A. Use the Google Cloud BigQuery component for Kubeflow Pipelines. Copy that component's URL, and use it to load the component into your pipeline. Use the component to execute queries against a BigQuery table.",
            "B. Use the Kubeflow Pipelines domain‐specific language to create a custom component that uses the Python BigQuery client library to execute queries.",
            "C. Use the BigQuery console to execute your query and then save the query results into a new BigQuery table.",
            "D. Write a Python script that uses the BigQuery API to execute queries against BigQuery. Execute this script as the first step in your pipeline in Kubeflow Pipelines."
            ],
            "answer": [
            "A. Use the Google Cloud BigQuery component for Kubeflow Pipelines. Copy that component's URL, and use it to load the component into your pipeline. Use the component to execute queries against a BigQuery table.",
            "D. Write a Python script that uses the BigQuery API to execute queries against BigQuery. Execute this script as the first step in your pipeline in Kubeflow Pipelines."
            ]
        },
        {
            "question": "You are a data scientist training a TensorFlow model with graph operations as operations that perform decoding and rounding tasks. You are using TensorFlow data transform to create data transformations and TFServing to serve your data. Your ML architect has asked you to set up MLOps and orchestrate the model serving only if data transformation is complete. Which of the following orchestrators can you choose to orchestrate your ML workflow? (Choose two.)",
            "options": [
            "A. Apache Airflow",
            "B. Kubeflow",
            "C. TFX",
            "D. Dataflow"
            ],
            "answer": ["A. Apache Airflow", "B. Kubeflow"]
        },
        {
            "question": "You are a data scientist working on creating an image classification model on Vertex AI. You are using Kubeflow to automate the current ML workflow. Which of the following options will help you set up the pipeline on Google Cloud with the least amount of effort?",
            "options": [
            "A. Set up Kubeflow Pipelines on GKE.",
            "B. Use Vertex AI Pipelines to set up Kubeflow ML pipelines.",
            "C. Set up Kubeflow Pipelines on an EC2 instance with autoscaling.",
            "D. Set up Kubeflow Pipelines using Cloud Run."
            ],
            "answer": ["B. Use Vertex AI Pipelines to set up Kubeflow ML pipelines."]
        },
        {
            "question": "As an ML engineer, you have written unit tests for a Kubeflow pipeline that require custom libraries. You want to automate the execution of unit tests with each new push to your development branch in Cloud Source Repositories. What is the recommended way?",
            "options": [
            "A. Write a script that sequentially performs the push to your development branch and executes the unit tests on Cloud Run.",
            "B. Create an event‐based Cloud Function when new code is pushed to Cloud Source Repositories to trigger a build.",
            "C. Using Cloud Build, set an automated trigger to execute the unit tests when changes are pushed to your development branch.",
            "D. Set up a Cloud Logging sink to a Pub/Sub topic that captures interactions with Cloud Source Repositories. Execute the unit tests using a Cloud Function that is triggered when messages are sent to the Pub/Sub topic."
            ],
            "answer": ["C. Using Cloud Build, set an automated trigger to execute the unit tests when changes are pushed to your development branch."]
        },
        {
            "question": "Your team is building a training pipeline on‐premises. Due to security limitations, they cannot move the data and model to the cloud. What is the recommended way to scale the pipeline?",
            "options": [
            "A. Use Anthos to set up Kubeflow Pipelines on GKE on‐premises.",
            "B. Use Anthos to set up Cloud Run to trigger training jobs on GKE on‐premises. Orchestrate all of the runs manually.",
            "C. Use Anthos to set up Cloud Run on‐premises to create a Vertex AI Pipelines job.",
            "D. Use Anthos to set up Cloud Run on‐premises to create a Vertex AI"
            ],
            "answer": ["A. Use Anthos to set up Kubeflow Pipelines on GKE on‐premises."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap12():
    st.title("Chapter 12: Model Monitoring, Tracking, and Auditing Metadata")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    "- **Understand model monitoring**. Understand the need to monitor the performance of the model after deployment. There are two main types of degradation: data drift and concept drift. Learn how to monitor continuously for these kinds of changes to input.""\n- **Learn logging strategies**. Logging after deployment is crucial to be able to keep track of the deployment, including the performance, as well as create new training data. Learn how to use logging in addition to monitoring the models in Vertex AI""\n- **Understand Vertex ML Metadata**. ML metadata helps you to track lineage of the models and other artifacts. Vertex ML Metadata is a managed solution for storing and accessing metadata on GCP. Learn the data model as well as the basic operations of creating and querying metadata."

    questions = [
        {
            "question": "You spend several months fine‐tuning your model and the model is performing very well in your evaluations based on test data. You have deployed your model, and over time you notice that the model accuracy is low. What happened and what should you do? (Choose two.)",
            "options": [
            "A. Nothing happened. There is only a temporary glitch.",
            "B. You need to enable monitoring to establish if the input data has drifted from the train/test data.",
            "C. Throw away the model and retrain with a higher threshold of accuracy.",
            "D. Collect more data from your input stream and use that to create training data, then retrain the model."
            ],
            "answer": ["B. You need to enable monitoring to establish if the input data has drifted from the train/test data.", "D. Collect more data from your input stream and use that to create training data, then retrain the model."]
        },
        {
            "question": "You spend several months fine‐tuning your model and the model is performing very well in your evaluations based on test data. You have deployed your model and it is performing well on real‐time data as well based on an initial assessment. Do you still need to monitor the deployment?",
            "options": [
            "A. It is not necessary because it performed very well with test data.",
            "B. It is not necessary because it performed well with test data and also on real‐time data on initial assessment.",
            "C. Yes. Monitoring the model is necessary no matter how well it might have performed on test data.",
            "D. It is not necessary because of cost constraints."
            ],
            "answer": ["C. Yes. Monitoring the model is necessary no matter how well it might have performed on test data."]
        },
        {
            "question": "Which of the following are two types of drift?",
            "options": [
            "A. Data drift",
            "B. Technical drift",
            "C. Slow drift",
            "D. Concept drift"
            ],
            "answer": ["A. Data drift", "D. Concept drift"]
        },
        {
            "question": "You trained a regression model to predict the longevity of a tree, and one of the input features was the height of the tree. When the model is deployed, you find that the average height of trees you are seeing is two standard deviations away from your input. What type of drift is this?",
            "options": [
            "A. Data drift",
            "B. Technical drift",
            "C. Slow drift",
            "D. Concept drift"
            ],
            "answer": ["A. Data drift"]
        },
        {
            "question": "You trained a classification model to predict fraudulent transactions and got a high F1 score. When the model was deployed initially, you had good results, but after a year, your model is not catching fraud. What type of drift is this?",
            "options": [
            "A. Data drift",
            "B. Technical drift",
            "C. Slow drift",
            "D. Concept drift"
            ],
            "answer": ["D. Concept drift"]
        },
        {
            "question": "When there is a difference in the input feature distribution between the training data and the data in production, what is this called?",
            "options": [
            "A. Distribution drift",
            "B. Feature drift",
            "C. Training‐serving skew",
            "D. Concept drift"
            ],
            "answer": ["C. Training‐serving skew"]
        },
        {
            "question": "When statistical distribution of the input feature in production data changes over time, what is this called in Vertex AI?",
            "options": [
            "A. Distribution drift",
            "B. Prediction drift",
            "C. Training‐serving skew",
            "D. Concept drift"
            ],
            "answer": ["B. Prediction drift"]
        },
        {
            "question": "You trained a classification model to predict the number of plankton in an image of ocean water taken using a microscope to measure the amount of plankton in the ocean. When the model is deployed, you find that the average number of plankton is an order of magnitude away from your training data. Later, you investigate this and find out it is because the magnification of the microscope was different in the training data. What type of drift is this?",
            "options": [
            "A. Data drift",
            "B. Technical drift",
            "C. Slow drift",
            "D. Concept drift"
            ],
            "answer": ["A. Data drift"]
        },
        {
            "question": "What is needed to detect training‐serving skew? (Choose two.)",
            "options": [
            "A. Baseline statistical distribution of input features in training data",
            "B. Baseline statistical distribution of input features in production data",
            "C. Continuous statistical distribution of features in training data",
            "D. Continuous statistical distribution of features in production data"
            ],
            "answer": [
            "A. Baseline statistical distribution of input features in training data",
            "D. Continuous statistical distribution of features in production data"
            ]
        },
        {
            "question": "What is needed to detect prediction drift? (Choose two.)",
            "options": [
            "A. Baseline statistical distribution of input features in training data",
            "B. Baseline statistical distribution of input features in production data",
            "C. Continuous statistical distribution of features in training data",
            "D. Continuous statistical distribution of features in production data"
            ],
            "answer": [
            "B. Baseline statistical distribution of input features in production data",
            "D. Continuous statistical distribution of features in production data"
            ]
        },
        {
            "question": "What is the distance score used for categorical features in Vertex AI?",
            "options": [
            "A. L‐infinity distance",
            "B. Count of the number of times the categorical value occurs over time",
            "C. Jensen‐Shannon divergence",
            "D. Normalized percentage of the time the categorical values differ"
            ],
            "answer": ["A. L‐infinity distance"]
        },
        {
            "question": "You deployed a model on an endpoint and enabled monitoring. You want to reduce cost. Which of the following is a valid approach?",
            "options": [
            "A. Periodically switch off monitoring to save money.",
            "B. Reduce the sampling rate to an appropriate level.",
            "C. Reduce the inputs to the model to reduce the monitoring footprint.",
            "D. Choose a high threshold so that alerts are not sent too often."
            ],
            "answer": ["B. Reduce the sampling rate to an appropriate level."]
        },
        {
            "question": "Which of the following are features of Vertex AI model monitoring? (Choose three.)",
            "options": [
            "A. Sampling rate: Configure a prediction request sampling rate.",
            "B. Monitoring frequency: Rate at which model’s inputs are monitored.",
            "C. Choose different distance metrics: Choose one of the many distance scores for each feature.",
            "D. Alerting thresholds: Set the threshold at which alerts will be sent."
            ],
            "answer": ["A. Sampling rate: Configure a prediction request sampling rate.", "B. Monitoring frequency: Rate at which model’s inputs are monitored.", "D. Alerting thresholds: Set the threshold at which alerts will be sent."]
        },
        {
            "question": "Which of the following is not a correct combination of model building and schema parsing in Vertex AI model monitoring?",
            "options": [
            "A. AutoML model with automatic schema parsing",
            "B. Custom model with automatic schema parsing with values in key‐value pairs",
            "C. Custom model with automatic schema parsing with values not in key‐value pairs",
            "D. Custom model with custom schema specified with values not in key‐value pairs"
            ],
            "answer": ["C. Custom model with automatic schema parsing with values not in key‐value pairs"]
        },
        {
            "question": "Which of the following is not a valid data type in the model monitoring schema?",
            "options": [
            "A. String",
            "B. Number",
            "C. Array",
            "D. Category"
            ],
            "answer": ["D. Category"]
        },
        {
            "question": "Which of the following is not a valid logging type in Vertex AI?",
            "options": [
            "A. Container logging",
            "B. Input logging",
            "C. Access logging",
            "D. Request‐response logging"
            ],
            "answer": ["B. Input logging"]
        },
        {
            "question": "How can you get a log of a sample of the prediction requests and responses?",
            "options": [
            "A. Container logging",
            "B. Input logging",
            "C. Access logging",
            "D. Request‐response logging"
            ],
            "answer": ["D. Request‐response logging"]
        },
        {
            "question": "Which of the following is a not a valid reason for using a metadata store?",
            "options": [
            "A. To compare the effectiveness of different sets of hyperparameters",
            "B. To track lineage",
            "C. To find the right proportion of train and test data",
            "D. To track downstream usage of artifacts for audit purposes"
            ],
            "answer": ["C. To find the right proportion of train and test data"]
        },
        {
            "question": "What is an artifact in a metadata store?",
            "options": [
            "A. Any piece of information on the metadata store",
            "B. The train and test dataset",
            "C. Any entity or a piece of data that was created by or can be consumed by an ML workflow",
            "D. A step in the ML workflow that can be annotated with runtime parameters"
            ],
            "answer": ["C. Any entity or a piece of data that was created by or can be consumed by an ML workflow"]
        },
        {
            "question": "Which of the following is not part of the data model in a Vertex ML metadata store?",
            "options": [
            "A. Artifact",
            "B. Workflow step",
            "C. Context",
            "D. Execution"
            ],
            "answer": ["B. Workflow step"]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap13():
    st.title("Chapter 13: Maintaining ML Solutions")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    '- **Understand MLOps maturity**. Learn different levels of maturity of MLOps and how it matches with the organizational goals. Know the MLOps architecture at the experimental phase, then a strategic phase where there is some automation, and finally a fully mature CI/CD‐inspired MLOps architecture.''\n- **Understand model versioning and retraining triggers**. A common problem faced in MLOps is knowing when to trigger new training. It could be based on model degradation as observed in model monitoring, or it could be time‐based. When retraining a model, learn how to add it as a new version or a new model.''\n- **Understand the use of feature store**. Feature engineering is an expensive operation, so the features generated using those methods are more useful if shared between teams. Vertex AI Feature Store is a managed service, and Feast is an open source feature store by Google.'

    questions = [
        {
            "question": "Which of the following is *not* one of the major steps in the MLOps workflow?",
            "options": [
            "A. Data processing, including extraction, analysis, and preparation",
            "B. Integration with third‐party software and identifying further use cases for similar models",
            "C. Model training, testing, and validation",
            "D. Deployment of the model, monitoring, and triggering retraining"
            ],
            "answer": ["B. Integration with third‐party software and identifying further use cases for similar models"]
        },
        {
            "question": "You are on a small ML team in a very old retail organization, and the organization is looking to start exploring machine learning for predicting daily sales of products. What level of MLOps would you implement in this situation?",
            "options": [
            "A. No MLOps, will build ML models ad hoc",
            "B. MLOps level 0",
            "C. MLOps level 1",
            "D. MLOps level 2"
            ],
            "answer": ["B. MLOps level 0"]
        },
        {
            "question": "You are a data scientist working as part of an ML team that has experimented with ML for its online fashion retail store. The models you build match customers to the right size/fit of clothes. Organization has decided to build this out, and you are leading this effort. What is the level of MLOps you would implement here?",
            "options": [
            "A. No MLOps, will build ML models ad hoc",
            "B. MLOps level 0",
            "C. MLOps level 1",
            "D. MLOps level 2"
            ],
            "answer": ["C. MLOps level 1"]
        },
        {
            "question": "You have been hired as an ML engineer to work in a large organization that works on processing photos and images. The team creates models to identify objects in photos, faces in photos, and the orientation of photos (to automatically turn) and also models to adjust the colors of photos. The organization is also experimenting with new algorithms that can automatically create images from text. What is the level of MLOps you would recommend?",
            "options": [
            "A. No MLOps, ad hoc because they are using new algorithms",
            "B. MLOps level 0",
            "C. MLOps level 1",
            "D. MLOps level 2"
            ],
            "answer": ["D. MLOps level 2"]
        },
        {
            "question": "What problems does MLOps level 0 solve?",
            "options": [
            "A. It is ad hoc building of models so it does not solve any problems.",
            "B. It automates training so building models is a repeatable process.",
            "C. Model training is manual but deployment is automated once there is model handoff.",
            "D. It is complete automation from data to deployment"
            ],
            "answer": ["C. Model training is manual but deployment is automated once there is model handoff."]
        },
        {
            "question": "Which of these statements is false regarding MLOps level 1 (strategic phase)?",
            "options": [
            "A. Building models becomes a repeatable process due to training automation.",
            "B. Model training is triggered automatically by new data.",
            "C. Trained models are automatically packaged and deployed.",
            "D. The pipeline is automated to handle new libraries and algorithms."
            ],
            "answer": ["D. The pipeline is automated to handle new libraries and algorithms."]
        },
        {
            "question": "You are part of an ML engineering team of a large organization that has started using ML extensively across multiple products. It is experimenting with different algorithms and even creating its own new ML algorithms. What should be its MLOps maturity level to be able to scale?",
            "options": [
            "A. Ad hoc is the only level that works for the organization because it is using custom algorithms.",
            "B. MLOps level 0.",
            "C. MLOps level 1.",
            "D. MLOps level 2."
            ],
            "answer": ["D. MLOps level 2."]
        },
        {
            "question": "In MLOps level 1 of maturity (strategic phase), what is handed off to the deployment?",
            "options": [
            "A. The model file",
            "B. The container containing the model",
            "C. The pipeline to train a model",
            "D. The TensorFlow or ML framework libraries"
            ],
            "answer": ["C. The pipeline to train a model"]
        },
        {
            "question": "In MLOps level 0 of maturity (tactical phase) what is handed off to the deployment?",
            "options": [
            "A. The model file",
            "B. The container containing the model",
            "C. The pipeline to train a model",
            "D. The TensorFlow or ML framework libraries"
            ],
            "answer": ["A. The model file"]
        },
        {
            "question": "What triggers building a new model in MLOps level 2?",
            "options": [
            "A. Feature store",
            "B. Random trigger",
            "C. Performance degradation from monitoring",
            "D. ML Metadata Store"
            ],
            "answer": ["C. Performance degradation from monitoring"]
        },
        {
            "question": "What should you consider when you are setting the trigger for retraining a model? (Choose two.)",
            "options": [
            "A. The algorithm",
            "B. The frequency of triggering retrains",
            "C. Cost of retraining",
            "D. Time to access data"
            ],
            "answer": ["B. The frequency of triggering retrains", "C. Cost of retraining"]
        },
        {
            "question": "What are reasonable policies to apply for triggering retraining from a model monitoring data? (Choose two.)",
            "options": [
            "A. The amount of prediction requests to a model",
            "B. Model performance degradation below a threshold",
            "C. Security breach",
            "D. Sudden drop in performance of the model"
            ],
            "answer": ["B. Model performance degradation below a threshold", "D. Sudden drop in performance of the model"]
        },
        {
            "question": "When you train or retrain a model, when do you deploy a new version (as opposed to deploy as a new model)?",
            "options": [
            "A. Every time you train a model, it is deployed as a new version.",
            "B. Only models that have been uptrained from pretrained models get a new version.",
            "C. Never create a new version, always a new model.",
            "D. Whenever the model has similar inputs and outputs and is used for the same purpose."
            ],
            "answer": ["D. Whenever the model has similar inputs and outputs and is used for the same purpose."]
        },
        {
            "question": "Which of the following are good reasons to use a feature store? (Choose two.)",
            "options": [
            "A. There are many features for a model.",
            "B. There are many engineered features that have not been shared between teams.",
            "C. The features created by the data teams are not available during serving time, and this is creating training/serving differences.",
            "D. The models are built on a variety of features, including categorical variables and continuous variables."
            ],
            "answer": ["B. There are many engineered features that have not been shared between teams.", "C. The features created by the data teams are not available during serving time, and this is creating training/serving differences."]
        },
        {
            "question": "Which service does Feast not use?",
            "options": [
            "A. BigQuery",
            "B. Redis",
            "C. Gojek",
            "D. Apache Beam"
            ],
            "answer": ["C. Gojek"]
        },
        {
            "question": "What is the hierarchy of the Vertex AI Feature Store data model?",
            "options": [
            "A. Featurestore ‐> EntityType ‐> Feature",
            "B. Featurestore ‐> Entity ‐> Feature",
            "C. Featurestore ‐> Feature ‐> FeatureValue",
            "D. Featurestore ‐> Entity ‐> FeatureValue"
            ],
            "answer": ["A. Featurestore ‐> EntityType ‐> Feature"]
        },
        {
            "question": "What is the highest level in the hierarchy of the data model of a Vertex AI Feature Store called?",
            "options": [
            "A. Featurestore",
            "B. Entity",
            "C. Feature",
            "D. EntityType"
            ],
            "answer": ["A. Featurestore"]
        },
        {
            "question": "You are working in a small organization and dealing with structured data, and you have worked on creating multiple high‐value features. Now you want to use these features for machine learning training and make these features available for real‐time serving as well. You are given only a day to implement a good solution for this and then move on to a different project. Which options work best for you?",
            "options": [
            "A. Store the features in BigQuery and retrieve using the BigQuery Python client.",
            "B. Create a Feature Store from scratch using BigQuery, Redis, and Apache Beam.",
            "C. Download and install open‐source Feast.",
            "D. Use Vertex AI Feature Store."
            ],
            "answer": ["D. Use Vertex AI Feature Store."]
        },
        {
            "question": "Which of these statements is false?",
            "options": [
            "A. Vertex AI Feature Store can ingest from BigQuery.",
            "B. Vertex AI Feature Store can ingest from Google Cloud Storage.",
            "C. Vertex AI Feature Store can even store images.",
            "D. Vertex AI Feature Store serves features with low latency."
            ],
            "answer": ["C. Vertex AI Feature Store can even store images."]
        },
        {
            "question": "Which of these statements is true?",
            "options": [
            "A. Vertex AI Feature Store uses a time‐series model to store all data.",
            "B. Vertex AI Feature Store cannot ingest from Google Cloud Storage.",
            "C. Vertex AI Feature Store can even store images.",
            "D. Vertex AI Feature Store cannot serve features with low latency."
            ],
            "answer": ["A. Vertex AI Feature Store uses a time‐series model to store all data."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def chap14():
    st.title("Chapter 14: BigQuery ML")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"

    st.subheader('Exam Essentials')
    '- **Understand BigQuery and ML**. Learn the history of BigQuery and the innovation of bringing machine learning into a data warehouse and to data analysis and anyone familiar with SQL. Learn how to train, predict, and provide model explanations using SQL.''\n- **Be able to explain the differences between BigQuery ML and Vertex AI and how they work together**. These services offer similar features but are designed for different users. BigQuery ML is designed for analysts and anyone familiar with SQL, and Vertex AI is designed for ML engineers. Learn the various different integration points that make it seamless to work between the two services.''\n- **Understand BigQuery design patterns**. BigQuery has elegant solutions to recurring problems in machine learning. Hashing, transforms, and serverless predictions are easy to apply to your ML pipeline.'
    
    questions = [
        {
            "question": "You work as part of a large data analyst team in a company that owns a global footwear brand. The company manufactures in South Asia and distributes all over the globe. Its sales were affected during the COVID-19 pandemic and so was distribution. Your team has been asked to forecast sales per country with new data about the spread of the illness and a plan for recovery. Currently your data is on‐prem and sales data comes from all over the world weekly. What will you use to forecast?",
            "options": [
            "A. Use Vertex AI AutoML Tables to forecast sales as this is a distributed case.",
            "B. User Vertex AI AutoML Tables with custom models (TensorFlow) because this is a special case due to COVID‐19.",
            "C. Use BigQuery ML, experiment with a TensorFlow model and DNN models to find the best results.",
            "D. Use BigQuery ML with ARIMA_PLUS, and use the BigQuery COVID‐19 public dataset for trends."
            ],
            "answer": ["D. Use BigQuery ML with ARIMA_PLUS, and use the BigQuery COVID‐19 public dataset for trends."]
        },
        {
            "question": "You are part of a startup that rents bicycles, and you want to predict the amount of time a bicycle will be used and the distance it will be taken based on current location and userid. You are part of a small team of data analysts, and currently all the data is sitting in a data warehouse. Your manager asks you to quickly create a machine learning model so that they can evaluate this idea. Your manager wants to show this prototype to the CEO to improve sales. What will you choose?",
            "options": [
            "A. Use a TensorFlow model on Vertex AI tables to predict time and distance.",
            "B. Use the advanced path prediction algorithm in Google Maps.",
            "C. Use BigQuery ML.",
            "D. Use a Vertex AI custom model to get better results because the inputs include map coordinates."
            ],
            "answer": ["C. Use BigQuery ML."]
        },
        {
            "question": "You are a data analyst for a large video sharing website. The website has thousands of users that provide 5‐star ratings for videos. You have been asked to provide recommendations per user. What would you use?",
            "options": [
            "A. Use BigQuery classification model_type.",
            "B. Use a Vertex AI custom model to build a collaborative filtering model and serve it online.",
            "C. Use the matrix factorization model in BigQuery ML to create recommendations using explicit feedback.",
            "D. Use Vertex AI AutoML for matrix factorization."
            ],
            "answer": ["C. Use the matrix factorization model in BigQuery ML to create recommendations using explicit feedback."]
        },
        {
            "question": "You are a data analyst and your manager gave you a TensorFlow SavedModel to use for a classification. You need to get some predictions quickly but don’t want to set up any instances or create pipelines. What would be your approach?",
            "options": [
            "A. Use BigQuery ML and choose TensorFlow as the model type to run predictions.",
            "B. Use Vertex AI custom models, and create a custom container with the TensorFlow SavedModel.",
            "C. TensorFlow SavedModel can only be used locally, so download the data onto a Jupyter Notebook and predict locally.",
            "D. Use Kubeflow to create predictions."
            ],
            "answer": ["A. Use BigQuery ML and choose TensorFlow as the model type to run predictions."]
        },
        {
            "question": "You are working as a data scientist in the finance industry and there are regulations about collecting and storing explanations for every machine learning prediction. You have been tasked to provide an initial machine learning model to classify good loans and loans that have defaulted. The model that you provide will be used initially and is expected to be improved further by a data analyst team. What is your solution?",
            "options": [
            "A. Use Kubeflow Pipelines to create a Vertex AI AutoML Table with explanations.",
            "B. Use Vertex AI Pipelines to create a Vertex AI AutoML Table with explanations and store them in BigQuery for analysts to work on.",
            "C. Use BigQuery ML, and select “classification” as the model type and enable explanations.",
            "D. Use Vertex AI AutoML Tables with explanations and store the results in BigQuery ML for analysts."
            ],
            "answer": ["C. Use BigQuery ML, and select “classification” as the model type and enable explanations."]
        },
        {
            "question": "You are a data scientist and have built extensive Vertex AI Pipelines which use Vertex AI AutoML Tables. Your manager is asking you to build a new model with data in BigQuery. How do you want to proceed?",
            "options": [
            "A. Create a Vertex AI pipeline component to download the BigQuery dataset to a GCS bucket and then run Vertex AI AutoML Tables.",
            "B. Create a new Vertex AI pipeline component to train BigQuery ML models on the BigQuery data.",
            "C. Create a Vertex AI pipeline component to execute Vertex AI AutoML by directly importing a BigQuery dataset.",
            "D. Create a schedule query to train a model in BigQuery."
            ],
            "answer": ["C. Create a Vertex AI pipeline component to execute Vertex AI AutoML by directly importing a BigQuery dataset."]
        },
        {
            "question": "You are a data scientist and have built extensive Vertex AI Pipelines which use Vertex AI AutoML Tables. Your manager is asking you to build a new model with a BigQuery public dataset. How do you want to proceed?",
            "options": [
            "A. Create a Vertex AI pipeline component to download the BigQuery dataset to a GCS bucket and then run Vertex AI AutoML Tables.",
            "B. Create a new Vertex AI pipeline component to train BigQuery ML models on the BigQuery data.",
            "C. Create a Vertex AI pipeline component to execute Vertex AI AutoML by directly importing the BigQuery public dataset.",
            "D. Train a model in BigQuery ML because it is not possible to access BigQuery public datasets from Vertex AI."
            ],
            "answer": ["C. Create a Vertex AI pipeline component to execute Vertex AI AutoML by directly importing the BigQuery public dataset."]
        },
        {
            "question": "You are a data scientist, and your team extensively uses Jupyter Notebooks. You are merging with the data analytics team, which uses only BigQuery. You have been asked to build models with new data that the analyst team created in BigQuery. How do you want to access it?",
            "options": [
            "A. Export the BigQuery data to GCS and then download it to the Vertex AI notebook.",
            "B. Create an automated Vertex AI pipeline job to download the BigQuery data to a GCS bucket and then download it to the Vertex AI notebook.",
            "C. Use Vertex AI managed notebooks, which can directly access BigQuery tables.",
            "D. Start using BigQuery console to accommodate the analysts."
            ],
            "answer": ["C. Use Vertex AI managed notebooks, which can directly access BigQuery tables."]
        },
        {
            "question": "You are a data scientist, and your team extensively uses Vertex AI AutoML Tables and pipelines. Your manager wants you to send the predictions of new test data to test for bias and fairness. The fairness test will be done by the analytics team that is comfortable with SQL. How do you want to access it?",
            "options": [
            "A. Export the test prediction data from GCS and create an automation job to transfer it to BigQuery for analysis.",
            "B. Move your model to BigQuery ML and create predictions there.",
            "C. Deploy the model and run a batch prediction on the new dataset to save in GCS and then transfer to BigQuery.",
            "D. Add the new data to your AutoML tables test set, and configure the Vertex AI tables to export test results to BigQuery."
            ],
            "answer": ["D. Add the new data to your AutoML tables test set, and configure the Vertex AI tables to export test results to BigQuery."]
        },
        {
            "question": "You are a data scientist, and your team extensively uses Vertex AI AutoML Tables and pipelines. Your manager wants you to send predictions to test for bias and fairness. The fairness test will be done by the analytics team that is comfortable with SQL. How do you want to access it?",
            "options": [
            "A. Export the test prediction data from GCS and create an automation job to transfer it to BigQuery for analysis.",
            "B. Move your model to BigQuery ML and create predictions there.",
            "C. Deploy the model and run a batch prediction on the new dataset to save in GCS and then transfer to BigQuery.",
            "D. Deploy the model and run a batch prediction on the new dataset to export directly to BigQuery."
            ],
            "answer": ["D. Deploy the model and run a batch prediction on the new dataset to export directly to BigQuery."]
        },
        {
            "question": "You are a data scientist, and your team extensively uses Vertex AI AutoML Tables and pipelines. Another team of analysts has built some highly accurate models on BigQuery ML. You want to use those models also as part of your pipeline. What is your solution?",
            "options": [
            "A. Run predictions in BigQuery and export the prediction data from BigQuery into GCS and then load it into your pipeline.",
            "B. Retrain the models on Vertex AI tables with the same data and hyperparameters.",
            "C. Load the models in the Vertex AI model repository and run batch predictions in Vertex AI.",
            "D. Download the model and create a container for Vertex AI custom models and run batch predictions."
            ],
            "answer": ["C. Load the models in the Vertex AI model repository and run batch predictions in Vertex AI."]
        },
        {
            "question": "You are a data analyst and working with structured data. You are exploring different machine learning options, including Vertex AI and BigQuery ML. You have found that your model accuracy is suffering because of a categorical feature (zipcode) that has high cardinality. You do not know if this feature is causing it. How can you fix this?",
            "options": [
            "A. Use the hashing function (ABS(MOD(FARM_FINGERPRINT(zipcode),buckets)) in BigQuery to bucketize.",
            "B. Remove the input feature and train without it.",
            "C. Don’t change the input as it affects accuracy.",
            "D. Vertex AI tables will automatically take care of this."
            ],
            "answer": ["A. Use the hashing function (ABS(MOD(FARM_FINGERPRINT(zipcode),buckets)) in BigQuery to bucketize."]
        },
        {
            "question": "You are a data analyst working with structured data in BigQuery and you want to perform some simple feature engineering (hashing, bucketizing) to improve your model accuracy. What are your options?",
            "options": [
            "A. Use the BigQuery TRANSFORM clause during CREATE_MODEL for your feature engineering.",
            "B. Have a sequence of queries to transform your data and then use this data for BigQuery ML training.",
            "C. Use Data Fusion to perform feature engineering and then load it into BigQuery.",
            "D. Build Vertex AI AutoML tables which can automatically take care of this problem."
            ],
            "answer": ["A. Use the BigQuery TRANSFORM clause during CREATE_MODEL for your feature engineering."]
        },
        {
            "question": "You are part of a data analyst team working with structured data in BigQuery but also considering using Vertex AI AutoML. Which of the following statements is wrong?",
            "options": [
            "A. You can run BigQuery ML models in Vertex AI AutoML tables.",
            "B. You can use BigQuery public datasets in AutoML tables.",
            "C. You can import data from BigQuery into AutoML.",
            "D. You can use SQL queries on Vertex AI AutoML tables."
            ],
            "answer": ["D. You can use SQL queries on Vertex AI AutoML tables."]
        },
        {
            "question": "Which of the following statements is wrong?",
            "options": [
            "A. You can run SQL in BigQuery through Python.",
            "B. You can run SQL in BigQuery through the CLI.",
            "C. You can run SQL in BigQuery through R.",
            "D. You can run SQL in BigQuery through Vertex AI."
            ],
            "answer": ["D. You can run SQL in BigQuery through Vertex AI."]
        },
        {
            "question": "You are training models on BigQuery but also use Vertex AI AutoML tables and custom models. You want flexibility in using data and models and want portability. Which of the following is a bad idea?",
            "options": [
            "A. Bring TensorFlow models into BigQuery ML.",
            "B. Use TRANSFORM functionality in BigQuery ML.",
            "C. Use BigQuery public datasets for training.",
            "D. Use Vertex AI Pipelines for automation."
            ],
            "answer": ["B. Use TRANSFORM functionality in BigQuery ML."]
        },
        {
            "question": "You want to standardize your MLOps using Vertex AI, especially AutoML Tables and Vertex AI Pipelines, etc., but some of your team is using BigQuery ML. Which of the following is incorrect?",
            "options": [
            "A. Vertex AI Pipelines will work with BigQuery.",
            "B. BigQuery ML models that include TRANSFORM can also be run on AutoML.",
            "C. BigQuery public datasets can be used in Vertex AI AutoML tables.",
            "D. You can use BigQuery and BigQuery ML through Python from Vertex AI managed notebooks."
            ],
            "answer": ["B. BigQuery ML models that include TRANSFORM can also be run on AutoML."]
        },
        {
            "question": "Which of these statements about BigQuery ML is incorrect?",
            "options": [
            "A. BigQuery ML supports both supervised and unsupervised models.",
            "B. BigQuery ML supports models for recommendation engines.",
            "C. You can control the various hyperparameters of a deep learning model like dropouts in BigQuery ML.",
            "D. BigQuery ML models with TRANSFORM clause can be ported to Vertex AI."
            ],
            "answer": ["D. BigQuery ML models with TRANSFORM clause can be ported to Vertex AI."]
        },
        {
            "question": "Which of these statements about comparing BigQuery ML explanations is incorrect?",
            "options": [
            "A. All BigQuery ML models provide explanations with each prediction.",
            "B. Feature attributions are provided both at the global level and for each prediction.",
            "C. The explanations vary by the type of model used.",
            "D. Not all models have global explanations."
            ],
            "answer": ["A. All BigQuery ML models provide explanations with each prediction."]
        },
        {
            "question": "You work as part of a large data analyst team in a company that owns hundreds of retail stores across the country. Their sales were affected due to bad weather. Currently your data is on‐prem and sales data comes from all across the country. What will you use to forecast sales using weather data?",
            "options": [
            "A. Use Vertex AI AutoML tables to forecast with previous sales data.",
            "B. User Vertex AI AutoML tables with a custom model (TensorFlow) and augment the data with weather data.",
            "C. Use BigQuery ML, and use the Wide‐and‐Deep model to forecast sales for a wide number of stores as well as deep into the future.",
            "D. Use BigQuery ML with ARIMA_PLUS, and use the BigQuery public weather dataset for trends."
            ],
            "answer": ["D. Use BigQuery ML with ARIMA_PLUS, and use the BigQuery public weather dataset for trends."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def bonus1():
    st.title("Bonus Exam 1")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"
    
    questions = [
        {
            "question": "Your company has used Google’s AutoML Tables to develop classification models. You have been asked by senior management to explain how the models work based on the features selected. You need to select a feature attribution method to help with the model explanation on nondifferentiable models. Which method will you choose?",
            "options": [
            "A. Feature attribution",
            "B. Sampled Shapley",
            "C. Integrated gradient",
            "D. XRAI (eXplanation with Ranked Area Integrals)"
            ],
            "answer": ["B. Sampled Shapley"]
        },
        {
            "question": "Your data science team has developed a model inference pipeline in Google Cloud. The pipeline is scheduled to run thrice per day and the data files are stored in Cloud Storage. The data now comes in more often due to high demand of products. How would you make sure the pipeline runs immediately when the new data is available with least effort? (Choose two.)",
            "options": [
            "A. Configure Cloud Functions with a Pub/Sub trigger on the Cloud Storage bucket to trigger the pipeline when new data is available.",
            "B. Create a Pub/Sub topic for Cloud Storage. Add code in your Cloud Run service with a Cloud Pub/Sb push to trigger a pipeline when new data is available in the storage bucket.",
            "C. Create a Cloud Storage polling application to retrain the pipeline based on new data.",
            "D. Create a Pub/Sub topic for Cloud Storage. Add code in your Cloud Run service with Cloud Pub/Sub polling logic to trigger the pipeline when new data is available in the storage bucket."
            ],
            "answer": [
            "A. Configure Cloud Functions with a Pub/Sub trigger on the Cloud Storage bucket to trigger the pipeline when new data is available.",
            "B. Create a Pub/Sub topic for Cloud Storage. Add code in your Cloud Run service with a Cloud Pub/Sb push to trigger a pipeline when new data is available in the storage bucket."
            ]
        },
        {
            "question": "Your company has adopted a multicloud strategy and started working on the Google Cloud platform. You have developed a forecast model with TensorFlow. How can you take advantage of Google Cloud while still being able to run the model training in other clouds with minimum or no code changes with the least effort?",
            "options": [
            "A. Create a custom container and train using Vertex AI custom training.",
            "B. Package the code as a zip file and use Vertex AI prebuilt container training.",
            "C. Use AutoML forecasting to train the model.",
            "D. Create a custom container, provision a deep learning container on Google Kubernetes Engine, and perform the training."
            ],
            "answer": ["A. Create a custom container and train using Vertex AI custom training."]
        },
        {
            "question": "You are a data scientist developing an ML model to detect pneumonia based on X‐ray images. You want to develop the model with minimal effort, but you don’t have enough X‐ray images to train a custom model. How would you train the model with least effort and time on GCP?",
            "options": [
            "A. Use data augmentation technique for the available images and train an AutoML Vision model.",
            "B. Train a TensorFlow model after augmenting the existing data using data augmentation techniques.",
            "C. Acquire more datasets from various sources and then train a custom CNN model.",
            "D. Train a custom CNN model with existing data."
            ],
            "answer": ["A. Use data augmentation technique for the available images and train an AutoML Vision model."]
        },
        {
            "question": "You are a data scientist working on a binary classification model to predict how likely a patient would get the flu if they have not been vaccinated in the past. After training the model, you find that the precision is lower than anticipated, so you decide to increase the classification threshold. How will increasing the classification threshold impact the other metrics?",
            "options": [
            "A. Precision will increase; recall will decrease or stay the same.",
            "B. Only recall will increase.",
            "C. Only precision will increase.",
            "D. Recall will increase; precision will stay the same."
            ],
            "answer": ["A. Precision will increase; recall will decrease or stay the same."]
        },
        {
            "question": "You are an ML lead engineer responsible for the MLOps process to deploy a machine learning model to production. The model requires ongoing updates to adapt to changes in the machine learning model and training pipeline. The team needs to respond quickly to these changes without manual steps, so they want to implement an ML continuous integration (CI/CD) process. Which of the following steps will they take with to set up the CI/CD pipeline in automated manner?",
            "options": [
            "A. New training data is available; run the training pipeline to train the model and evaluate the model.",
            "B. Developer commits model/pipeline code. Trigger a build on Cloud Build to build the pipeline. Deploy and run the training pipeline to train a new model version.",
            "C. Developer commits model/pipeline code. Deploy and run the training pipeline to train a new model version. Trigger a build on Cloud Build to build the pipeline.",
            "D. New training data is available. Deploy and run the training pipeline to train a new model version. Trigger a build on Cloud Build to build the pipeline."
            ],
            "answer": ["B. Developer commits model/pipeline code. Trigger a build on Cloud Build to build the pipeline. Deploy and run the training pipeline to train a new model version."]
        },
        {
            "question": "You are MLOps practitioner, and you are orchestrating ML pipelines in production in Google Cloud. The basic components for the training pipeline has five steps: (1) Data storage (BigQuery), (2) Data Extraction, Validation and Data Transformation, (3) Model training (Vertex AI Platform Training), (4) Model evaluation/validation, (5) Model serving (Vertex AI Platform Prediction). You need to decide on the right Google service to implement steps 2 and 4 and a solution to orchestrate the pipeline. Which of the following solutions is the automated way to run the pipeline?",
            "options": [
            "A. Use Dataprep for steps 2 and 3 and Kubeflow Pipelines to orchestrate the pipeline.",
            "B. Use Data Fusion for steps 2 and 3 and Cloud Composer to orchestrate the pipeline.",
            "C. Use Dataflow for steps 2 and 3 and Kubeflow Pipelines to orchestrate the pipeline.",
            "D. Use Dataproc for steps 2 and 3 and Cloud Composer to orchestrate the pipeline."
            ],
            "answer": ["C. Use Dataflow for steps 2 and 3 and Kubeflow Pipelines to orchestrate the pipeline."]
        },
        {
            "question": "You are training a deep learning model to predict the likelihood of getting a job after earning a graduate degree. After doing some initial data analysis on the training dataset, you notice that one numeric feature is distributed relatively uniformly but contains a few extreme outliers. This feature could affect your model's performance and stability and you want to take steps to avoid this. What can you do to ensure that this feature does not negatively affect the training stability and model performance?",
            "options": [
            "A. Use a clipping strategy when normalizing the data during the data preparation step.",
            "B. Use L2 regularization during model training.",
            "C. Use the Sigmoid activation function while training the layers.",
            "D. Use the linear scaling strategy when normalizing the data during the data preparation step."
            ],
            "answer": ["A. Use a clipping strategy when normalizing the data during the data preparation step."]
        },
        {
            "question": "You are a data scientist of a Fortune 500 company, and your team has developed a model using a deep neural network in TensorFlow. The model uses hundreds of features and has over 200 layers. The model performs well with the training dataset, but when it is tested against a test dataset, the model shows poor performance. The team leader asks you to help troubleshoot the issue and improve the model performance. What do you do to improve the model performance? (Choose two.)",
            "options": [
            "A. Use dropout to reduce the number of layers.",
            "B. Use L1 regularization to reduce the number of features.",
            "C. Split the dataset into a random split for model training and testing.",
            "D. Use the data augmentation technique to generate more data."
            ],
            "answer": ["A. Use dropout to reduce the number of layers.", "B. Use L1 regularization to reduce the number of features."]
        },
        {
            "question": "You are a data scientist working for a team that uses the Google Cloud Vertex AI platform and TensorFlow to implement a new model into production. The team expects the preprocessing step to involve the following operations: (1) Cleansing data: Correct data that is invalid or missing; (2) Feature tuning: Normalizing numeric data and clipping outliers in numeric data. The statistics generated during training to normalize data need to be available when data preparation happens on some input data for prediction. Which of these solutions can the team choose to ensure that the computed statistics needed for data preparation are available with the exported model used for serving predictions?",
            "options": [
            "A. Transform data using Dataflow and store the statistics needed to prepare data at prediction time in Cloud SQL.",
            "B. Transform data using Dataflow and the TensorFlow Transform library.",
            "C. Transform data using Dataflow and the TensorFlow Data Validation library.",
            "D. Transform data using BigQuery and store the statistics needed to prepare data at prediction time."
            ],
            "answer": ["B. Transform data using Dataflow and the TensorFlow Transform library."]
        },
        {
            "question": "You are an ML engineer who has created an ML training pipeline using Dataflow and TFX in Google Cloud. Your company states that the model training pipelines must be able to run both on premises and in Goggle Cloud. You need to select an orchestration tool to run the data and ML pipelines. Which tool should you use to satisfy this requirement?",
            "options": [
            "A. Kubeflow Pipelines",
            "B. Vertex AI pipelines",
            "C. TFX (TensorFlow Extended)",
            "D. Cloud Composer"
            ],
            "answer": ["A. Kubeflow Pipelines"]
        },
        {
            "question": "You are developing a machine learning model to categorize recipes uploaded to the site as either vegan, vegetarian, or non‐vegetarian. The training data is labeled using one‐hot encoding, and you plan to use TensorFlow on the Google Cloud Vertex AI platform to train your custom model. Which of these loss functions should you choose to train this model?",
            "options": [
            "A. Categorical cross‐entropy",
            "B. Sparse cross‐entropy",
            "C. Mean squared error",
            "D. Binary cross‐entropy"
            ],
            "answer": ["A. Categorical cross‐entropy"]
        },
        {
            "question": "You are working on a classification model to predict if an image is a picture of a plane, a helicopter, or a fighter jet. You are planning to use TensorFlow and build your model using a deep neural network (DNN). The output of the model should be the probability that the image is a plane, a helicopter, or a fighter jet. How many output nodes should your model include, and which activation function should you choose for the output layer?",
            "options": [
            "A. Choose an output layer with three output nodes and the softmax activation function.",
            "B. Choose an output layer with three output nodes and the reLU activation function.",
            "C. Choose an output layer with one output node and the reLU activation function.",
            "D. Choose an output layer with two output nodes and the softmax activation function."
            ],
            "answer": ["A. Choose an output layer with three output nodes and the softmax activation function."]
        },
        {
            "question": "You are a data scientist of a research firm training a machine learning model using a deep neural network (DNN) to predict the house prices in New York City. You notice that the generalization curve shows that the loss for the training data continues to decrease gradually with the number of iterations. In contrast, the loss for the validation data set initially decreases but begins to increase with more iterations. What is likely causing this behavior, and what can you do to address this issue?",
            "options": [
            "A. Overfitting; retrain the model with L2 regularization.",
            "B. Underfitting; retrain the model with early stopping.",
            "C. Overfitting; retrain the model with a large dataset and less iterations.",
            "D. Underfitting; retrain the model with additional layers."
            ],
            "answer": ["A. Overfitting; retrain the model with L2 regularization."]
        },
        {
            "question": "You are an ML engineer using Google Colab Jupyter Notebooks to experiment with a new binary classification model. During the model development process, you would like to use an interactive visual tool to compare two models. Also, the tool you use must be capable of evaluating different fairness policies. Which of the following Google AI tools would be most appropriate for your investigation?",
            "options": [
            "A. Vertex AI interactive shell",
            "B. What‐If Tool within the notebook",
            "C. Vertex AI TensorBoard Profiler",
            "D. IPython Magics for BigQuery"
            ],
            "answer": ["B. What‐If Tool within the notebook"]
        },
        {
            "question": "You are collecting data from surveys to train a model in Google Cloud. This data is in the form of unstructured text and is stored in Google Cloud Storage. This data includes personally identifiable information. What Google service should the team choose to help ensure that personal user information is protected during model training?",
            "options": [
            "A. IAM (identity and access management)",
            "B. Cloud DLP (Data Loss Prevention) API",
            "C. Secret Manager",
            "D. VPC Service Control"
            ],
            "answer": ["B. Cloud DLP (Data Loss Prevention) API"]
        },
        {
            "question": "You are training a classification model to detect fraudulent transactions of a banking system. Your training dataset includes 10,000 examples: 100 are labeled as “fraud” and 9,900 are marked “not fraud.” After you train the model, the accuracy is about 98 percent. Moreover, the generalization curve shows the loss for the training and validation data gradually decreasing and leveling off with the number of iterations. However, you find in production that the model is not identifying any fraudulent activity. What is likely causing this problem in production?",
            "options": [
            "A. Overfitting",
            "B. Class imbalance",
            "C. Early stopping",
            "D. Regularization"
            ],
            "answer": ["B. Class imbalance"]
        },
        {
            "question": "You are using a Google Vertex AI platform notebook to develop a new machine learning model. Your dataset is in BigQuery, and you plan to write some preprocessing code to prepare the data using Python before running your training experiments. You need to decide the best way to load the data. Which of the following approaches should you use to load your data from BigQuery into your notebook?",
            "options": [
            "A. Export data from BigQuery to Cloud Storage and then read in the data from Cloud Storage.",
            "B. Use BigQuery connector to provide read access from within the AI platform notebook.",
            "C. Use the BigQuery client library for Python to load data into the notebook as a DataFrame.",
            "D. Create a Cloud Dataflow job to read data from BigQuery tables."
            ],
            "answer": ["C. Use the BigQuery client library for Python to load data into the notebook as a DataFrame."]
        },
        {
            "question": "Your company is using Google Vertex AI for managing machine learning across data science teams. Your security team wants to view what activities the data science team is doing. Also, there is an admin team that wants to manage all resources and grant permissions. Which of the following identity and access management (IAM) roles can help set up the requirements? (Choose two.)",
            "options": [
            "A. Grant the security team the Vertex AI viewer role.",
            "B. Grant the admin team the Vertex AI administrator role.",
            "C. Grant the security team the custom IAM role.",
            "D. Grant the admin team the custom IAM role."
            ],
            "answer": ["A. Grant the security team the Vertex AI viewer role.", "B. Grant the admin team the Vertex AI administrator role."]
        },
        {
            "question": "You are working on a machine learning model that uses PyTorch. You are migrating the current models in the project from on‐premise to the Google Cloud Vertex AI platform. There is no code dependency requirement for the model training. Which of the following options should you recommend for this machine learning project to run on the Vertex AI platform with the least effort?",
            "options": [
            "A. Use an existing prebuilt PyTorch container for Vertex AI training.",
            "B. Create a Vertex AI custom container‐training job.",
            "C. Migrate your project to TensorFlow and perform the training with a TensorFlow container.",
            "D. Migrate your data and use AutoML."
            ],
            "answer": ["A. Use an existing prebuilt PyTorch container for Vertex AI training."]
        },
        {
            "question": "You are a data scientist working on a linear regression machine learning model to predict the price of cars. The color of the cars is saved in the dataset as categorical data with possible values such as “blue”, “black”, “silver”, and “white”. As you prepare the dataset before training, you need to decide how best to handle this nonnumeric feature. Which of the following approaches should you choose to prepare the data for model training?",
            "options": [
            "A. Use label encoding or integer encoding such as blue=1, black=2, silver=3, and so on.",
            "B. Use an embedding layer.",
            "C. Use bucketing.",
            "D. Use one‐hot encoding to transform the nonnumeric features."
            ],
            "answer": ["D. Use one‐hot encoding to transform the nonnumeric features."]
        },
        {
            "question": "You are a security ML engineer working for a hedge fund. You work on large datasets that have a lot of private information that cannot be distributed and disclosed. You are asked to replace sensitive data with specific surrogate characters. Which of the following techniques do you think is best to use?",
            "options": [
            "A. Masking",
            "B. Replacement",
            "C. Tokenization",
            "D. Format‐preserving encryption"
            ],
            "answer": ["A. Masking"]
        },
        {
            "question": "You are a data scientist and you are going to develop an ML model with Python. This model will be embedded in your client applications. You are setting up authentication for the project to use Vertex AI client libraries for training and prediction. What are you going to do? (Choose two.)",
            "options": [
            "A. Create a service account key to authenticate your application.",
            "B. Set the environment variable named GOOGLE_APPLICATION_CREDENTIALS.",
            "C. Create a custom IAM role.",
            "D. Use single‐user settings in Jupyter Workbench."
            ],
            "answer": ["A. Create a service account key to authenticate your application.", "B. Set the environment variable named GOOGLE_APPLICATION_CREDENTIALS."]
        },
        {
            "question": "You are a data scientist working on building a model to predict house prices using Vertex AI. Your feature set performs well during training but in production the quality of inference is degraded. You perform various checks, but the model seems to be perfectly fine. Finally, when you look at the input data, you notice that the frequency distributions have changed for a specific feature. Which GCP service can be helpful for you to manage features in a more organized way and avoid the skew?",
            "options": [
            "A. Hyperparameter tuning",
            "B. Model monitoring",
            "C. Feature Store",
            "D. Model registry"
            ],
            "answer": ["C. Feature Store"]
        },
        {
            "question": "You are a data scientist who is working on a deep neural network model with TensorFlow to classify defective products. You are using a GPU for training your models. However, your training is taking longer than usual and you need to debug the performance of your models. Which is the best solution that you can adopt?",
            "options": [
            "A. TFTrace",
            "B. TFProfiler",
            "C. What‐If Tool",
            "D. None of the above"
            ],
            "answer": ["B. TFProfiler"]
        },
        {
            "question": "You develop, train, and deploy several ML models with TensorFlow. You use data in Parquet format and need to manage it both in input and output. You want to create an optimized input pipeline to increase the performance of training sessions using tf.Data API. Which of the techniques can help? (Choose three.)",
            "options": [
            "A. Prefetching",
            "B. Caching",
            "C. Parallelizing data",
            "D. None of the above"
            ],
            "answer": ["A. Prefetching", "B. Caching", "C. Parallelizing data"]
        },
        {
            "question": "You are a data scientist who has been building a model to host in an environment to predict car sales. You have been asked by management to explain which online or batch endpoint you would have to create on the GCP Vertex AI platform. What are two of the characteristics of using an online prediction rather than a batch prediction? (Choose two.)",
            "options": [
            "A. With online prediction, you get near real‐time response with no latency.",
            "B. The prediction endpoint is up and running 24/7.",
            "C. The cost to maintain the online endpoint is minimum.",
            "D. You can set up model monitoring to the online prediction."
            ],
            "answer": ["A. With online prediction, you get near real‐time response with no latency.", "B. The prediction endpoint is up and running 24/7."]
        },
        {
            "question": "Your team is designing a fraud detection system for a major bank. The data will be stored in real time with some statistical aggregations. An ML model will be periodically trained for outlier detection. The ML model will issue the probability of fraud for each transaction. You need to pick Google Cloud services to build a pipeline that will stream the data, perform data transformation, and train a model. Which three services would you pick? (Choose three.)",
            "options": [
            "A. Cloud Pub/Sub Lite",
            "B. Dataflow",
            "C. Vertex AI Pipelines",
            "D. BigQuery ML"
            ],
            "answer": ["A. Cloud Pub/Sub Lite", "B. Dataflow", "C. Vertex AI Pipelines"]
        },
        {
            "question": "Your company runs an e‐commerce site. You produced static deep learning models with TensorFlow that have been in production for some time. Initially, they gave you excellent results, but gradually the accuracy has progressively decreased. You retrained the models with the new data and solved the problem. At this point, you want to automate the process using the Google Cloud environment. Which of these solutions allows you to reach your goal with the least effort?",
            "options": [
            "A. Kubeflow on Google Kubernetes Engine",
            "B. TFX and the Vertex AI platform",
            "C. Apache Airflow and the Vertex AI platform",
            "D. Cloud Dataflow and BigQuery"
            ],
            "answer": ["B. TFX and the Vertex AI platform"]
        },
        {
            "question": "You are building a linear regression model with a very large subset of features. You want to simplify the model to make it more efficient and faster. Your first goal is to synthesize the features without losing the information content that comes from them. Which of the below algorithm you will choose?",
            "options": [
            "A. Data augmentation",
            "B. GANs (generative adversarial networks)",
            "C. PCA",
            "D. L2 regularization"
            ],
            "answer": ["C. PCA"]
        },
        {
            "question": "You are training a set of models that should be simple, using regression techniques. During training, your models seem to work. But the tests are giving unsatisfactory results. You discover that you have missing data. You need a tool that helps you analyze the missing data. Which GCP product would you choose?",
            "options": [
            "A. Cloud Dataprep",
            "B. Cloud Dataproc",
            "C. Cloud Dataflow",
            "D. Cloud Composer"
            ],
            "answer": ["A. Cloud Dataprep"]
        },
        {
            "question": "You are a data scientist in an insurance firm building a new model with a small dataset. However, the model accuracy is not satisfactory because your data is dirty and needs to be modified. You have various fields that have no value or report NaN. Which of the following techniques you would *not* choose to clean data without affecting model performance?",
            "options": [
            "A. Use another ML model for guessing missing values.",
            "B. Compute mean/median for numeric values and replace missing values.",
            "C. Delete the rows or columns with missing values such as null or NaN.",
            "D. Replace missing values with the most frequent category."
            ],
            "answer": ["C. Delete the rows or columns with missing values such as null or NaN."]
        },
        {
            "question": "You are a data scientist and you trained a text classification model using TensorFlow. You have downloaded the saved model for TFServing. The model predict request is as follows:\n\njson dumps({'signature_name': f,serving_default', 'instances': [['a', 'b'], ['c', 'd'], ['e', 'f']]})\nWhat can be the possible shape of the tensor in the input SignatureDef?",
            "options": [
            "A. shape: (‐1, 2)",
            "B. shape: (‐1, 3)",
            "C. shape: (‐1, 5)",
            "D. shape: (‐1, 4)"
            ],
            "answer": ["A. shape: (‐1, 2)"]
        },
        {
            "question": "You are a data scientist working on training a machine learning model with 500,000 labeled images of products. You are using the Vertex AI platform and taking advantage of TPU accelerators to improve training times. You want to make sure reading in the large volume of images does not create a bottleneck during training. Which of the following solutions can help reduce bottleneck issues with data ingestion?",
            "options": [
            "A. Store images in Cloud Storage and read images during training using Dataproc.",
            "B. Store images in Cloud Storage and read images using tf.data.dataset for training.",
            "C. Convert images as TFRecords and store them in Cloud Storage and read images using the tf.data.TFRecordDataset format for training.",
            "D. Compress the images as a zip file and read using tf.data.dataset for training."
            ],
            "answer": ["C. Convert images as TFRecords and store them in Cloud Storage and read images using the tf.data.TFRecordDataset format for training."]
        },
        {
            "question": "You are training a set of models that should be simple, using regression techniques. You need to ingest real time data as an input to the trained model. You need a tool that helps you cope with it. Which GCP product would you choose?",
            "options": [
            "A. Cloud Dataprep",
            "B. Cloud Dataproc",
            "C. Cloud Dataflow",
            "D. Cloud Composer"
            ],
            "answer": ["C. Cloud Dataflow"]
        },
        {
            "question": "What is the distance score used for numerical features in Vertex AI?",
            "options": [
            "A. L‐infinity distance",
            "B. Difference in the mean of the two distributions",
            "C. Difference in the median of the two distributions",
            "D. Jensen‐Shannon divergence"
            ],
            "answer": ["D. Jensen‐Shannon divergence"]
        },
        {
            "question": "You built a regression model on AutoML and you want to deploy the model and enable monitoring. Which of these steps is *not* valid?",
            "options": [
            "A. Create an endpoint, and deploy the model.",
            "B. Configure Vertex AI model monitoring with sampling rate, monitoring frequency and alert thresholds.",
            "C. Create a schema for the input values to provide to model monitoring, without which it cannot parse.",
            "D. Enable model monitoring for both skew and drift."
            ],
            "answer": ["C. Create a schema for the input values to provide to model monitoring, without which it cannot parse."]
        },
        {
            "question": "You are building an ML model to predict the share price of a petroleum refining company. You company is betting on the stock going down and will lose a lot of money if the particular stock goes up. You have been asked to build a regression model (and not a forecasting model) to predict the value of a share based on daily parameters like price of a gallon of gas, price of diesel, and the crude oil price. What metric would you use for this regression model?",
            "options": [
            "A. Mean absolute error (MAE)",
            "B. Root‐mean‐square error (RMSE)",
            "C. Root‐mean‐squared logarithmic error (RMSLE)",
            "D. Mean absolute percentage error (MAPE)"
            ],
            "answer": ["C. Root‐mean‐squared logarithmic error (RMSLE)"]
        },
        {
            "question": "You work for a clothing retail store and have a website where you get millions of hits on a daily basis. However, management thinks the revenue from the website is too low. You are asked to improve the revenue using recommendations. You have decided to use Recommendations AI to build models and show recommendations during check out.\n\nWhat type of model in Recommendations AI would you choose?",
            "options": [
            "A. “Others you may like”",
            "B. “Frequently bought together”",
            "C. “Similar items”",
            "D. “Recommended for you”"
            ],
            "answer": ["B. “Frequently bought together”"]
        },
        {
            "question": "You are training a large deep learning TensorFlow model using Keras (more than 200 GB) on a dataset that has a matrix in which most values are zeros. You used only Keras inbuilt operators and no custom operations. You wanted to use TPUs so you have optimized the training loop to not have any I/O operations. Which of the following hardware accelerator options is best suited in this case?",
            "options": [
            "A. A single TPU host because the model was created using Keras.",
            "B. Use a TPU Pod because the size of the model is very large.",
            "C. Use TPU v4 slice that is the appropriate size and shape for the use case.",
            "D. Use a GPU."
            ],
            "answer": ["D. Use a GPU."]
        },
        {
            "question": "You are building a mobile app to classify different types of plants using images of leaves from trees and plants. You have thousands of images of leaves. How do you go about training and using the model?",
            "options": [
            "A. Use Vertex AI AutoML to train an image classification model, with AutoML Edge as the method. Use a Coral.ai device that has Edge TPU and deploy the model on that device.",
            "B. Use Vertex AI AutoML to train an image classification model, with AutoML Edge as the method. Create an Android app using ML Kit and deploy the model to the edge device.",
            "C. Use Vertex AI AutoML to train an object detection model with AutoML Edge as the method. Use a Coral.ai device that has Edge TPU and deploy the model on that device.",
            "D. Use Vertex AI AutoML to train an image segmentation model with AutoML Edge as the method. Create an Android app using ML Kit and deploy the model to the edge device."
            ],
            "answer": ["B. Use Vertex AI AutoML to train an image classification model, with AutoML Edge as the method. Create an Android app using ML Kit and deploy the model to the edge device."]
        },
        {
            "question": "Your company sent out a survey to all your customers and received feedback from millions of them. After initial analysis, you realized they either talked about product P1 or P2 but not both. Now your manager wants you to split the feedback into two buckets and find the sentiment for each product. The executives are waiting for this data, so you have only a few days to complete this. Luckily you already have about 2,000 of these comments already labeled with the product type (P1 or P2). What will be your approach?",
            "options": [
            "A. Use Vertex AI AutoML to train a text classification model and a sentiment analysis model on the text data. Run a batch prediction on the data.",
            "B. Use a Vertex AI custom container to train a deep learning TensorFlow model for sentiment analysis, which also highlights the entities that are salient.",
            "C. Use the Google Natural Language API to find the sentiment of the feedback. Separately create a Vertex AI AutoML text classification model to classify the text into P1 or P2. Combine these two predictions for each feedback and present it.",
            "D. Use Vertex AI AutoML to train a logistic regression model for predicting the sentiment and a separate AutoML for predicting the class P1 or P2 and then combine the two predictions."
            ],
            "answer": ["C. Use the Google Natural Language API to find the sentiment of the feedback. Separately create a Vertex AI AutoML text classification model to classify the text into P1 or P2. Combine these two predictions for each feedback and present it."]
        },
        {
            "question": "You trained a deep learning model using a Vertex AI custom model and deployed the model on a Vertex AI endpoint and enabled monitoring. You don’t expect changes to inputs values and want to reduce monitoring costs. Which of the following is a valid approach?",
            "options": [
            "A. Switch off monitoring to save costs.",
            "B. Choose a high threshold so that alerts are not sent too often.",
            "C. Add more models to this endpoint to split the monitoring costs between the models.",
            "D. Reduce the sampling rate to an appropriate level."
            ],
            "answer": ["D. Reduce the sampling rate to an appropriate level."]
        },
        {
            "question": "In the data model of Vertex ML Metadata, what is a context?",
            "options": [
            "A. An entity or a piece of data that was created by or can be consumed by ML workflow",
            "B. A group of artifacts and executions that can be queried",
            "C. A step in an ML workflow that can be annotated with runtime parameters",
            "D. The schema to be used by particular types of data, like artifact or execution"
            ],
            "answer": ["B. A group of artifacts and executions that can be queried"]
        },
        {
            "question": "You have trained a deep learning model, and it performs well on the test and validation data. After deployment, you have enabled Vertex AI model monitoring and notice that the accuracy is dropping slowly and realize that the model will have to be retrained at some point. Your organization has matured to MLOps level 2 and wants to automatically trigger retraining. What are two valid retraining trigger policies? (Choose two.)",
            "options": [
            "A. Whenever accuracy drops below an absolute threshold, you trigger retraining.",
            "B. Don’t wait for model accuracy to drop, but continuously retrain models and keep deploying whenever the model is ready.",
            "C. Whenever the accuracy drops suddenly. For example, it is drops more than 2 percent a day, you trigger retraining.",
            "D. Don’t retrain. This will be done manually."
            ],
            "answer": ["A. Whenever accuracy drops below an absolute threshold, you trigger retraining.", "C. Whenever the accuracy drops suddenly. For example, it is drops more than 2 percent a day, you trigger retraining."]
        },
        {
            "question": "You are an ML engineer working for a clothing retailer that has hundreds of stores. You are asked to build a model to predict weekly sales for each store. Your company has several data science and data engineering teams and they are creating many useful features that you think you may need. The features are not being effectively used. How can you improve this situation?",
            "options": [
            "A. Move all the data to BigQuery and use BigQuery ML.",
            "B. Use a Vertex AI managed dataset.",
            "C. Use Vertex AI Workbench so all data scientists are on the same platform.",
            "D. Use Vertex AI Feature Store to help centrally store the features and share between teams."
            ],
            "answer": ["D. Use Vertex AI Feature Store to help centrally store the features and share between teams."]
        },
        {
            "question": "You are a data analyst in a large fast‐food chain business. You have terabytes of data on the orders, restaurant details, employee details, and inventory details. Your team is very comfortable with SQL and now wants to use machine learning to improve efficiency of the restaurants and reduce food waste. What is the best ML tool for your team?",
            "options": [
            "A. Vertex AI AutoML Tables",
            "B. BigQuery ML",
            "C. Vertex AI Workbench",
            "D. Vertex AI Feature"
            ],
            "answer": ["B. BigQuery ML"]
        },
        {
            "question": "You are part of a team of data analysts and your team has trained model in a TensorFlow SavedModel format to use for a classification. You need to get batch predictions for about 1 billion datapoints quickly but don’t want to set up any instances or create pipelines. What would be your approach?",
            "options": [
            "A. Use Vertex AI custom models and create a custom container with the TensorFlow SavedModel format.",
            "B. Use BigQuery ML and choose TensorFlow as model type to run predictions.",
            "C. TensorFlow SavedModel can only be used locally, so download the data onto a Jupyter notebook and predict locally.",
            "D. Use Kubeflow to create predictions."
            ],
            "answer": ["B. Use BigQuery ML and choose TensorFlow as model type to run predictions."]
        },
        {
            "question": "You are a data scientist that loves to use Jupyter notebooks. You have joined a large data team that exclusively uses BigQuery. You have been given access to large datasets (1 TB) in BigQuery to build models and run batch predictions to the others. How do you want to access the data?",
            "options": [
            "A. Use Vertex AI Workbench to run your Jupyter notebooks, and export the BigQuery data to your workbench instance and build models there.",
            "B. Create a Vertex AI pipeline job to download the BigQuery data to GCS bucket and download it to Workbench Jupyter Notebook, where you can use Python.",
            "C. Use Vertex AI managed notebooks that can directly access BigQuery tables. You can also use the BigQuery magic command to directly run BigQuery SQL from notebooks.",
            "D. Use the BigQuery web console to align with your team."
            ],
            "answer": ["C. Use Vertex AI managed notebooks that can directly access BigQuery tables. You can also use the BigQuery magic command to directly run BigQuery SQL from notebooks."]
        },
        {
            "question": "You are part of a team of data scientists that use Vertex AI and some of the data analysts are building models in BigQuery ML. Vertex AI is being used for AutoML tables and building custom models for structured data only. You want flexibility in using data and models and want portability. Which of the following should *not* be done?",
            "options": [
            "A. Use TRANSFORM functions in BigQuery ML.",
            "B. Bring TensorFlow models into BigQuery ML.",
            "C. Use BigQuery public datasets for training.",
            "D. Use Vertex AI pipelines for automation."
            ],
            "answer": ["A. Use TRANSFORM functions in BigQuery ML."]
        }
    ]

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

def bonus2():
    st.title("Bonus Exam 2")
    "Material from [Official Google Cloud Certified Professional Machine Learning Engineer Study Guide](https://www.wiley.com/en-us/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461)"
    
    questions = [
        {
            "question": "Your company has used Vertex AI custom model training to develop classification models. You are performing data cleaning and you plotted a normal distribution curve that is not symmetric. What could be the reason for this? (Choose two.)",
            "options": [
            "A. Data skew",
            "B. Outliers in the data",
            "C. Missing data",
            "D. Imbalanced data"
            ],
            "answer": ["A. Data skew", "B. Outliers in the data"]
        },
        {
            "question": "Your data science team has developed a model in Google Cloud. Your training dataset has age, sex, salary, and zip code as a feature. Your model is having difficulty converging and is giving importance to features having a wider range, such as age. Which of the following techniques will help?",
            "options": [
            "A. Scaling",
            "B. Covariance",
            "C. Clipping",
            "D. Box plot"
            ],
            "answer": ["A. Scaling"]
        },
        {
            "question": "Your company has adopted a multicloud strategy and started working on the Google Cloud platform. You have developed a forecast model with TensorFlow. Your dataset is 5 TB and you need to validate these large datasets. Which of the following approaches can help set up the data validation pipeline with the least effort?",
            "options": [
            "A. Use Dataflow to run TensorFlow Data Validation.",
            "B. Use Dataflow to run TensorFlow Transform.",
            "C. Set up TFX in a virtual machine and run TensorFlow Data Validation.",
            "D. Set up TFX in a virtual machine and run (TensorFlow Transform."
            ],
            "answer": ["A. Use Dataflow to run TensorFlow Data Validation."]
        },
        {
            "question": "You are a data scientist developing an ML model to classify flowers based on species. Your data has missing values and your team has asked you to come up with a model for beta testing. You do not have time to remove missing data. Which of the following algorithms might work with your current dataset? (Choose two.)",
            "options": [
            "A. Matrix factorization",
            "B. Naive Bayes",
            "C. K‐nearest neighbors",
            "D. Logistic regression"
            ],
            "answer": ["B. Naive Bayes", "C. K‐nearest neighbors"]
        },
        {
            "question": "You are a data scientist working on a binary classification model to predict how likely a patient will get the flu if he has not been vaccinated in the past. After training the model, you want to minimize false positives. How will that impact the other metrics?",
            "options": [
            "A. Raise the classification threshold.",
            "B. Lower the classification threshold.",
            "C. Increase recall.",
            "D. Decrease precision."
            ],
            "answer": ["A. Raise the classification threshold."]
        },
        {
            "question": "You are an ML lead engineer responsible for a financial firm. You have to create a model that will be deployed on websites to detect fraud transactions. You have the dataset from your team, but it does not have a lot of fraud examples to train your model. You trained a model using Google Cloud AutoML. Which metric you would use improve your predictions?",
            "options": [
            "A. Precision",
            "B. Log loss",
            "C. The area under the precision‐recall (AUC PR) curve value",
            "D. The area under the curve receiver operating characteristic (AUC ROC) curve value"
            ],
            "answer": ["C. The area under the precision‐recall (AUC PR) curve value"]
        },
        {
            "question": "You are a junior data scientist working on building a model that needs to predict how crowded the street‐based location of people is at a certain time of the day. You have features such as curbside, market, and time of the day. Which of the following sets of features or feature crosses could learn relationships between location, market, and time of the day?",
            "options": [
            "A. Two one‐hot encoded feature crosses: [location (curbside) X time of the day] and [location (market) X time of the day]",
            "B. Three separate features: [market], [curbside], [timeoftheday]",
            "C. One feature cross: [market X curbside X timeoftheday]",
            "D. One‐hot encoded feature cross: [market X curbside X timeoftheday]"
            ],
            "answer": ["D. One‐hot encoded feature cross: [market X curbside X timeoftheday]"]
        },
        {
            "question": "You are training a deep learning model to predict the likelihood of getting a job after earning a graduate degree. After doing some initial data analysis on the training data set, you notice that feature has columns with different ranges. What can you do to ensure this feature does not negatively affect the training stability and model performance?",
            "options": [
            "A. Use normalization and bucketing for data preparation.",
            "B. Use L2 regularization during model training.",
            "C. Use the Sigmoid activation function while training the layers.",
            "D. Use PCA and tSNE algorithms on your model."
            ],
            "answer": ["A. Use normalization and bucketing for data preparation."]
        },
        {
            "question": "You are a data scientist of a Fortune 500 company and your team has developed a model using a deep neural network in TensorFlow. The model uses hundreds of features and has over 200 layers. You want to reduce bottleneck with data loading and speed up your model training process. What technique you can use? (Choose two.)",
            "options": [
            "A. Use tf.data.Dataset.prefetch transformation while reading data.",
            "B. Use TensorFlow Transform from the TFX library.",
            "C. Use TensorFlow Data Validation from the TFX library.",
            "D. Use tf.data.Dataset.interleave transformation, which parallelizes the data reading."
            ],
            "answer": [
            "A. Use tf.data.Dataset.prefetch transformation while reading data.",
            "D. Use tf.data.Dataset.interleave transformation, which parallelizes the data reading."
            ]
        },
        {
            "question": "You are a data scientist working for a team that uses the Google Cloud Vertex AI platform and TensorFlow to implement a sales forecasting model into production. The sales data is stored in BigQuery as a tabular dataset. You have to implement the TensorFlow model to forecast sales data in BigQuery. How would you do this with the least effort?",
            "options": [
            "A. Use a custom TensorFlow model with BigQuery ML.",
            "B. Use Vertex AI prediction to host the model. Use App Engine to read data from BigQuery and perform prediction by calling the Vertex AI endpoint.",
            "C. Use the model registry to import the trained model. Use Vertex AI batch prediction with input dataset source as a BigQuery table.",
            "D. Use AutoML forecasting to train the model."
            ],
            "answer": ["A. Use a custom TensorFlow model with BigQuery ML."]
        },
        {
            "question": "You are an ML engineer for a media and gaming applications company. You have to build an efficient data storage and ML pipeline. The requirements are submillisecond latency when users are accessing the online application where the model is hosted. You also need to store intermediate data for a real‐time data pipeline for creating input features Which data storage will be best suited for intermediate storage?",
            "options": [
            "A. Memorystore",
            "B. Bigtable",
            "C. Datastore",
            "D. Hard disk"
            ],
            "answer": ["A. Memorystore"]
        },
        {
            "question": "You are an ML Lead engineer responsible for the MLOps process to deploy a machine learning model to production. You are using a TFX library to create your ML pipeline steps (data prep, train, and tune). You have implemented some pipeline workflows using Cloud Dataflow. However, it is difficult to configure, monitor, and maintain defined pipelines and workflows. You also need to track the model lineage, which is an ask from management. What is the recommended approach?",
            "options": [
            "A. Use Vertex AI Pipelines.",
            "B. Use Cloud Composer.",
            "C. Install Kubeflow Pipelines on GKE and use Kubeflow to orchestrate your model training with TensorFlow.",
            "D. Use TFX pipelines."
            ],
            "answer": ["A. Use Vertex AI Pipelines."]
        },
        {
            "question": "You are an ML customer engineer working on architecting a system for a financial firm. Your team has hired a data scientist to deploy a fraud detection model. You have a requirement to notify other systems to take action when a potentially fraudulent transaction is identified by the model in near real time. You team has recently migrated to Google Cloud Platform. How would you design this system with the least effort?",
            "options": [
            "A. Deploy your model on Vertex AI online prediction, process the fraud prediction events using Cloud Dataflow, and notify to other systems using Pub/Sub.",
            "B. Deploy your model on Vertex AI online prediction, process the fraud prediction events using Cloud Dataflow, and poll the predictions in Datastore (noSQLStore).",
            "C. Deploy your model on Google Kubernetes Engine, process the fraud events using Cloud Dataflow, and poll the predictions in Datastore.",
            "D. Run your model on Vertex AI batch prediction and notify to other systems using Pub/Sub."
            ],
            "answer": ["A. Deploy your model on Vertex AI online prediction, process the fraud prediction events using Cloud Dataflow, and notify to other systems using Pub/Sub."]
        },
        {
            "question": "You are an AI/ML customer engineer at a research firm. Your team has trained a TensorFlow deep learning model with a large batch size to predict the house prices in the New York City in Google Cloud Platform. The requirement to host this model for online predictions is that there is low latency while serving the model. You have deployed the model after training on GPU. However, your serving and model latency is not improving as the model is very large. Which of the following recommendations will improve this? (Choose two.)",
            "options": [
            "A. Build a smaller model and use cloud accelerators such as TPU for training.",
            "B. Precompute predictions in an offline batch‐scoring job, and cache the predictions.",
            "C. Retrain the model with a large dataset and fewer iterations.",
            "D. Retrain the model with additional layers."
            ],
            "answer": [
            "A. Build a smaller model and use cloud accelerators such as TPU for training.",
            "B. Precompute predictions in an offline batch‐scoring job, and cache the predictions."
            ]
        },
        {
            "question": "You are an ML engineer collecting data from surveys to train a model in Google Cloud. This data is in the form of unstructured text and is stored in Google Cloud Storage. Your management wants you to set up best practices with the ML training pipeline. They want to track lineage of the data with model training. They also want to label this data using a managed service and perform text classification using AutoML NLP. What is the recommended way to do this with the least effort?",
            "options": [
            "A. Use a Vertex managed dataset and Vertex data labeling.",
            "B. Use Vertex Feature Store and Vertex data labeling.",
            "C. Use no managed dataset and Vertex data labeling.",
            "D. Use Vertex Feature Store and a Vertex managed dataset."
            ],
            "answer": ["A. Use a Vertex managed dataset and Vertex data labeling."]
        },
        {
            "question": "Your team is using Vertex AI Feature Store to store all the features required for training models. You are a security engineer and you have been asked by management to grant access to read only a few features to the production team and access to read all features in Feature Store to the development team. Can use these two levels of granularity to customize permissions. How would you configure this using Google Cloud Identity and access management? (Choose two.)",
            "options": [
            "A. Grant permissions to particular feature stores by using a resource‐level policy for the production team.",
            "B. Grant read permission to all feature stores by setting a project‐level policy for development team.",
            "C. Grant permissions to particular feature stores by using a resource‐level policy for development team.",
            "D. Grant read permission to all feature stores by setting a project‐level policy for production team."
            ],
            "answer": [
            "A. Grant permissions to particular feature stores by using a resource‐level policy for the production team.",
            "B. Grant read permission to all feature stores by setting a project‐level policy for development team."
            ]
        },
        {
            "question": "You are assigned as a chief data scientist for a group of hospitals that are participating in the same clinical trial. The data that an individual hospital collects about patients is not shared outside the hospital. You are working on building a collaborative machine learning model using the datasets in the hospitals. What technique you would use to securely build this model without sharing the data between hospitals.?",
            "options": [
            "A. Differential privacy",
            "B. Federated learning",
            "C. Format preserving encryption",
            "D. Tokenization"
            ],
            "answer": ["B. Federated learning"]
        },
        {
            "question": "You are an ML architect for an online banking firm. Your team has recently migrated to Google Cloud. Your credit card transactions data is stored in Cloud Storage buckets, BigQuery tables, and Datastore. Your security team has asked you to identify sensitive data from these data sources. You created a Cloud DLP job to scan content for sensitive data. You also have new data coming in small batches every morning. How you would scan the sensitive content from the new data?",
            "options": [
            "A. You can trigger a DLP scan job by using Cloud Functions every time a file is uploaded to Cloud Storage.",
            "B. If you turn on data profiling, Cloud DLP automatically scans all BigQuery tables and columns across the entire organization, individual folders, and projects.",
            "C. You can use Dataflow to trigger a DLP job to de‐identify data.",
            "D. Create a Cloud Dataproc job to trigger a DLP job to de‐identify sensitive data."
            ],
            "answer": ["A. You can trigger a DLP scan job by using Cloud Functions every time a file is uploaded to Cloud Storage."]
        },
        {
            "question": "Your company is using Google Vertex AI for managing machine learning across data science teams. There is auditing that needs to happen in your organization. Your admin team has been asked to grant an external auditor a role to view all activities the data science team is doing. Which of the following identity and access management (IAM) roles can help set up the requirements?",
            "options": [
            "A. Vertex AI viewer role",
            "B. Vertex AI administrator role",
            "C. Custom IAM role",
            "D. Custom IAM permissions"
            ],
            "answer": ["A. Vertex AI viewer role"]
        },
        {
            "question": "You are a data scientist working on a machine learning model that uses TensorFlow to train large images. The neural network you are training has 500 layers. Your model training in local machine is taking days to complete. Your teammate suggested doing data parallel training because your team got access to GPUs and TPUs on Google Cloud Platform. You used synchronous training with tf.distribute.Strategy to train on multiple GPUs. However, you are finding it harder to scale and workers are staying idle at times. What should you do to improve your training time?",
            "options": [
            "A. Train your model with TF parameter server distributed training.",
            "B. Train your model with “All reduce sync strategy” on TPU.",
            "C. Split your model layers on multiple machines (model parallelism).",
            "D. Migrate your data and use AutoML."
            ],
            "answer": ["A. Train your model with TF parameter server distributed training."]
        },
        {
            "question": "You are a data scientist working on training a CNN model using an image dataset. You are doing hyperparameter tuning such as batch size and learning rate. However, your model is still taking longer than usual to complete training. When you debug the model, you are getting an out of memory error. Which hyperparameter you can tune to avoid this?",
            "options": [
            "A. Decrease the batch size.",
            "B. Increase the learning rate.",
            "C. Increase epoch.",
            "D. Increase the number of batches for training."
            ],
            "answer": ["A. Decrease the batch size."]
        },
        {
            "question": "You are a data scientist working in a transportation agency. You have a pretrained model to recognize cars. Now your management has asked you to train the model to identify trucks having a company logo. The organization does not have a large data example for truck detection. Which of the following techniques do you think is best to use?",
            "options": [
            "A. Use the transfer learning technique.",
            "B. Use the semi‐supervised learning technique.",
            "C. Use the data augmentation technique to generate data and then train a model.",
            "D. Use the data parallel approach while training the model."
            ],
            "answer": ["A. Use the transfer learning technique."]
        },
        {
            "question": "You are a data scientist and you are going to develop an ML model intended to detect fraud for a large bank. Some fraud you know about, but other instances of fraud are slipping by without your knowledge. You have a small, labeled dataset provide by the management team. You asked the management for labeled data but the management has no labeled data. You can label the dataset with the fraud instances you’re aware of, but the rest of your data is unlabeled. Which of the following techniques will help you build a model in this scenario?",
            "options": [
            "A. L1 regularization",
            "B. Transfer learning",
            "C. Data augmentation",
            "D. Semi‐supervised learning"
            ],
            "answer": ["D. Semi‐supervised learning"]
        },
        {
            "question": "You are a data scientist working on building a model to predict house prices using Vertex AI. You trained a neural network model with hyperparameters such as learning rate, epoch, and batch size. The dataset provided to you is small. The model you have trained is not converging and it is bouncing around. You decreased the learning rate, which resulted in reducing the training loss. Now your management team wants you to further fine‐tune your model so that your training loss is minimum. Which of the following options can help? (Choose two.)",
            "options": [
            "A. Increase the learning rate.",
            "B. Increase the depth and width of your layers.",
            "C. Use cross‐validation or bootstrapping.",
            "D. Dropout layers of neural network."
            ],
            "answer": [
            "B. Increase the depth and width of your layers.",
            "C. Use cross‐validation or bootstrapping."
            ]
        },
        {
            "question": "You are a data scientist who is working on a deep neural network model with TensorFlow to classify defective products. You are using GPU for training your models and your model has 100 layers. During training, the gradients for the lower layers has become very small. When the gradients vanish toward 0 for the lower layers, these layers train very slowly or they do not train at all. Which of the following technique can help?",
            "options": [
            "A. Use ReLU activation function.",
            "B. Use batch normalization.",
            "C. Decrease the learning rate.",
            "D. Use the Sigmoid activation function."
            ],
            "answer": ["A. Use ReLU activation function."]
        },
        {
            "question": "You are a data scientist who has been building a model to host in an environment to predict the car sales. Your management has decided to migrate all the ML projects to the Vertex AI platform. You data is in the form of text, videos, and images stored on a local drive. You want to use AutoML on Vertex AI for quick and fast development. Which of the following data store you would choose for this?",
            "options": [
            "A. Vertex AI managed datasets",
            "B. Google Cloud Storage",
            "C. BigQuery",
            "D. Vertex AI Feature Store"
            ],
            "answer": ["A. Vertex AI managed datasets"]
        },
        {
            "question": "You are a data scientist who has been building a model to predict inventory sales. Your models are in PyTorch and you want to use Vertex AI prebuilt containers for training your model. How would you package your existing code for training? (Choose two.)",
            "options": [
            "A. Create a root folder with setup.py and a trainer folder with task.py (training code), which is the entry point for Vertex AI Training jobs.",
            "B. Upload your training code as Python source distribution to a Cloud Storage bucket.",
            "C. Create a root folder. Then create a Dockerfile and a folder named Trainer. In that Trainer folder you need to create task.py (your training code).",
            "D. Create a custom container and training file."
            ],
            "answer": [
            "A. Create a root folder with setup.py and a trainer folder with task.py (training code), which is the entry point for Vertex AI Training jobs.",
            "B. Upload your training code as Python source distribution to a Cloud Storage bucket."
            ]
        },
        {
            "question": "Your team is designing a fraud detection system for a major bank using a TensorFlow model. You are planning to use Vertex AI custom training for training the model and to install custom libraries needed to run the model. You want to set up hyperparameter tuning for the custom trained model. What is the best way to do this?",
            "options": [
            "A. Create a Cron task to set up multiple training jobs with different hyperparameters.",
            "B. Install the cloud-ml hypertune Python package in your Dockerfile. Add hyperparameter tuning code in main() of the python file and add arguments. Build and push the container to the registry. Configure a hyperparameter tuning job using a training pipeline.",
            "C. Install the cloud-ml hypertune Python package in your Dockerfile. Add hyperparameter tuning code in main() of python file and add arguments. Configure a hyperparameter tuning job using a custom job.",
            "D. Install the cloud-ml hypertune Python package in your Dockerfile. Build and push the container to the registry. Configure a hyperparameter tuning job using a custom job."
            ],
            "answer": ["B. Install the cloud-ml hypertune Python package in your Dockerfile. Add hyperparameter tuning code in main() of the python file and add arguments. Build and push the container to the registry. Configure a hyperparameter tuning job using a training pipeline."]
        },
        {
            "question": "You are a data scientist working on an image classification model that is trained to predict whether a given image contains a dog or a cat. If you request predictions from this model on a new set of images, then you receive a prediction for each image (“dog” or “cat”). Your management team is setting up security practices and want you to explain each prediction for your model. You recently started using Vertex AI AutoML classification to classify these images. Which technique can you use for explanations for each image?",
            "options": [
            "A. Integrated gradients",
            "B. XRAI (eXplanation with Ranked Area Integrals)",
            "C. SHAP",
            "D. Vertex example‐based explanations"
            ],
            "answer": ["A. Integrated gradients"]
        },
        {
            "question": "You are building a TensorFlow model for image classification with a very large subset of features. You want to simplify the model to make it more efficient and faster. You need to understand which feature contributed to the predictions. You recently started using the Vertex AI platform for training your model. Which of the following options can help set up explanations in your model? (Choose three.)",
            "options": [
            "A. Use the Explainable AI SDK’s save_model_with_metadata() method to infer your model’s inputs and outputs and save this explanation metadata with your model.",
            "B. Load the model into the Explainable AI SDK using load_model_from_local_path().",
            "C. Call explain() with instances of data, and visualize the feature attributions.",
            "D. Load the model into the Explainable AI SDK using load_model_from_global_path()."
            ],
            "answer": [
            "A. Use the Explainable AI SDK’s save_model_with_metadata() method to infer your model’s inputs and outputs and save this explanation metadata with your model.",
            "B. Load the model into the Explainable AI SDK using load_model_from_local_path().",
            "C. Call explain() with instances of data, and visualize the feature attributions."
            ]
        },
        {
            "question": "You are a data scientist working on building a recommendation model. Your management has asked you to build a pipeline for the model that will recommend similar products given the attributes of the products that a customer is currently viewing. The attribute feature has feature columns such as customer id stored in BigQuery. The customer click‐through data is near real time based .Your model should recommend similar product based on this customer click through data. How would you set up this feature lookup pipeline.",
            "options": [
            "A. Process the data from BigQuery using Cloud Dataflow. Precompute and store the features to a datastore such as Cloud Firestore for feature lookup.",
            "B. Aggregate the output using streaming Dataflow in Cloud Bigtable for real‐time lookup.",
            "C. Process the data from BigQuery using Cloud Dataproc. Precompute and store the features to a datastore for feature lookup.",
            "D. Aggregate the output using streaming Dataproc in Cloud Bigtable for real‐time lookup."
            ],
            "answer": ["A. Process the data from BigQuery using Cloud Dataflow. Precompute and store the features to a datastore such as Cloud Firestore for feature lookup."]
        },
        {
            "question": "You are a data scientist in an insurance firm and you are building a new model with a small dataset. You have deployed this model using a Vertex AI prediction endpoint. You do not have a critical business and you need endpoints up and running for a few hours in a day or during weekends. How would you design this with the least effort?",
            "options": [
            "A. With Vertex AI managed notebooks, execute and schedule a Vertex prediction job using Jupyter Notebook.",
            "B. Use event‐driven Cloud Functions and Cloud Pub/Sub to trigger Vertex AI endpoint creation.",
            "C. Use Cloud Build with Cloud Run to schedule the pipeline.",
            "D. Use a Cloud Scheduler Cron job to schedule your prediction serving using Vertex AI prediction."
            ],
            "answer": ["D. Use a Cloud Scheduler Cron job to schedule your prediction serving using Vertex AI prediction."]
        },
        {
            "question": "You are an ML engineer who has created ML training pipelines using PyTorch and the MXNet framework in Google Cloud. You need to select an orchestration tool to run the ML pipelines. Which tools should you use to satisfy this requirement?",
            "options": [
            "A. Build the pipelines using Kubeflow Pipelines and deploy on GKE.",
            "B. Build the pipelines using Kubeflow Pipelines and deploy using Vertex AI Pipelines.",
            "C. Use TFX (TensorFlow Extended).",
            "D. Build the pipelines using Kubeflow Pipelines and deploy using Cloud Composer."
            ],
            "answer": ["A. Build the pipelines using Kubeflow Pipelines and deploy on GKE."]
        },
        {
            "question": "You are a data scientist working on training a machine learning model with 500,000 labeled images of products. Your company is working as a hybrid cloud. You are migrating to the Vertex AI platform on Google Cloud. Your data is stored in an Amazon S3 bucket and Azure Blob Storage blobs. You need some data from both and need to perform some joins in the data before using it for ML training. Which tools should you use to satisfy this requirement with the least effort?",
            "options": [
            "A. Use the LOAD DATA SQL statement with BigQuery Omni.",
            "B. Use the BigQuery data transfer service to move data.",
            "C. Download data from S3 and Azure and upload to Google Cloud Storage.",
            "D. Use Cloud Functions to collect data from S3 and Azure using APIs."
            ],
            "answer": ["A. Use the LOAD DATA SQL statement with BigQuery Omni."]
        },
        {
            "question": "You are training a set of models that should be simple to train, using regression techniques and TensorFlow. You started building the ML pipeline using TFX. You have the following requirements: (1) ingests and optionally splits the input dataset, (2) calculates statistics for the dataset, (3) detects anomalies and missing values in the dataset. Which of the following TFX libraries would you use? (Choose three.)",
            "options": [
            "A. ExampleGen",
            "B. StatisticsGen",
            "C. ExampleValidator",
            "D. SchemaGen"
            ],
            "answer": ["A. ExampleGen", "B. StatisticsGen", "C. ExampleValidator"]
        },
        {
            "question": "You are a machine learning expert working in a company that is trying to identify objects in orbit around the sun. You have been tasked with estimating the size (mass) of asteroids based on blurry images. You have been given a dataset of millions of images of asteroids labeled with estimated mass. What kind of machine learning problem is this?",
            "options": [
            "A. Unsupervised classification",
            "B. Supervised classification",
            "C. Unsupervised regression",
            "D. Supervised regression"
            ],
            "answer": ["D. Supervised regression"]
        },
        {
            "question": "You work in a regulated industry where there are laws about logging each time there is access to data and related cloud assets for future audit purposes. As a machine learning engineer, you are using Vertex AI to train and deploy models. You needed some help so you reached out to Google Support for resolution. Some of the Google’s personnel accessed your account to help you out. Does this affect your compliance and do you have do anything about it?",
            "options": [
            "A. This does not affect compliance. Google personnel are part of the cloud, so they are exempt.",
            "B. It does affect your compliance. To be compliant you should enable Access Transparency logs.",
            "C. This affects compliance, and you need to enable Audit logs.",
            "D. This affects compliance, so be compliant provide a key to a new service account for Google personnel to access and monitor activities using that key."
            ],
            "answer": ["B. It does affect your compliance. To be compliant you should enable Access Transparency logs."]
        },
        {
            "question": "You work for a startup that is creating an Android App to identify insects from photos. The company wants to build a quick prototype. You have a large, labeled dataset of over 1 million photos of thousands of insects. You are given one week to build a model. What would be your approach?",
            "options": [
            "A. One week is too short to build a model because of large number of classes.",
            "B. Use Vertex AI AutoML to create an image classification model.",
            "C. Create a custom TensorFlow model using Keras, and use Vertex AI training to train it.",
            "D. Use BigQuery ML train an AutoML model."
            ],
            "answer": ["B. Use Vertex AI AutoML to create an image classification model."]
        },
        {
            "question": "You are using TensorFlow to build an image classification model. When you call the model.predict() function, you get the error “Python inputs incompatible with input_signature.” What could be the problem?",
            "options": [
            "A. The security signature embedded in the model does not match with model security signature.",
            "B. You are using a “signature” dataset that is causing errors.",
            "C. The train code and predict code is using different versions of TensorFlow.",
            "D. The input values to the model.predict() functions do not match with the input used during training."
            ],
            "answer": ["D. The input values to the model.predict() functions do not match with the input used during training."]
        },
        {
            "question": "You retrained a model for better accuracy on the same data and want to add it to the model registry and deploy. Which of the following are best practices? (Choose two.)",
            "options": [
            "A. Register this as a new model under a new name, and deploy.",
            "B. Since this model is compatible with the previous versions, you can add this as the latest version of the existing model.",
            "C. Add metadata to this model to show that this was built on the same training data, with details about model architecture.",
            "D. Delete the previous model because it is not needed anymore."
            ],
            "answer": [
            "B. Since this model is compatible with the previous versions, you can add this as the latest version of the existing model.",
            "C. Add metadata to this model to show that this was built on the same training data, with details about model architecture."
            ]
        },
        {
            "question": "You are a data analyst in a large office supplies store with thousands of business customers. You have been asked to classify the customers in a customer database into segments to see patterns of usage. Which of the following is a good approach?",
            "options": [
            "A. Use the BigQuery ML k‐means clustering algorithm to cluster the customers into segments.",
            "B. Use Vertex AI Auto ML tables classification algorithm to classify the customers into different segments.",
            "C. Use the BigQuery ML classification algorithm to classify the customers into different segments.",
            "D. Use Scikit k‐means on Vertex AI to classify the customers into segments."
            ],
            "answer": ["A. Use the BigQuery ML k‐means clustering algorithm to cluster the customers into segments."]
        },
        {
            "question": "You are a data analyst in the financial industry and are building models to predict if a customer will default on next month’s mortgage payment. Your company is legally required to log every model and explain the prediction. You have used BigQuery ML to build a model but you do not get explanations from your model. What do you need to do? (Choose two.)",
            "options": [
            "A. Move out of BigQuery because BigQuery ML does not support model explanations.",
            "B. When training the model in BigQuery ML, set enable_global_explain to TRUE.",
            "C. Enable the correct transforms when you make predictions to get the matching explanations.",
            "D. Download the global explanations for each model you build by using ML.GLOBAL_EXPLAIN."
            ],
            "answer": [
            "B. When training the model in BigQuery ML, set enable_global_explain to TRUE.",
            "D. Download the global explanations for each model you build by using ML.GLOBAL_EXPLAIN."
            ]
        },
        {
            "question": "Which of the following statements about interoperability between BigQuery ML and Vertex AI is false?",
            "options": [
            "A. You cannot access BigQuery public datasets from Vertex AI.",
            "B. You can create Vertex AI datasets from a BigQuery tables.",
            "C. You cannot export BigQuery models if you used the TRANSFORM clause.",
            "D. Vertex AI Model Monitoring data is stored in BigQuery."
            ],
            "answer": ["A. You cannot access BigQuery public datasets from Vertex AI."]
        },
        {
            "question": "Your company has created a Vertex AI Feature Store to store features that will be used by many machine learning teams. What is the correct hierarchy of data in Vertex AI Feature Store?",
            "options": [
            "A. Featurestore ‐> EntityType ‐> Feature",
            "B. Featurestore ‐> Feature ‐> Entity",
            "C. Featurestore ‐> EntityType ‐> Entity",
            "D. Feature ‐> EntityType ‐> Entity"
            ],
            "answer": ["A. Featurestore ‐> EntityType ‐> Feature"]
        },
        {
            "question": "In Vertex AI Model Monitoring, there are three types of schema formats. Which of the following is not a valid type?",
            "options": [
            "A. Object",
            "B. Array",
            "C. Number",
            "D. String"
            ],
            "answer": ["C. Number"]
        },
        {
            "question": "In Vertex ML Metadata, what \"is a step in a machine learning workflow and can be annotated with runtime parameters”?",
            "options": [
            "A. Context",
            "B. Artifact",
            "C. Execution",
            "D. Event"
            ],
            "answer": ["C. Execution"]
        },
        {
            "question": "You are an ML engineer in a retail fashion store that wants to improve the user experience on its website as well. Currently it gets tens of thousands of visitors every day and collects customer’s browsing history. The company wants to engage the customer more to provide an immersive experience. You have decided to use Recommendations AI (Retail AI) for providing recommendations. Recommendations AI has different types of models. Which model will you choose for this business purpose?",
            "options": [
            "A. “Others you may like”",
            "B. “Frequently bought together”",
            "C. “Similar items”",
            "D. “Recommended for you”"
            ],
            "answer": ["A. “Others you may like”"]
        },
        {
            "question": "You are a business owner in an international transport broker company. Your company has hundreds of documents about customs clearance for the United States that need to be translated to more than 50 languages. You notice that some terminology in the documents are not common English words, so this needs to handled carefully. How would you approach this problem?",
            "options": [
            "A. Use Translate AutoML to translate documents into the target language and to handle these special words.",
            "B. Use Pretrained Translate Advanced version with Glossary using Python API.",
            "C. Use Translation Hub with a Glossary to handle special words.",
            "D. Use Google Translate Basic version."
            ],
            "answer": ["C. Use Translation Hub with a Glossary to handle special words."]
        },
        {
            "question": "You are a machine learning engineer working in a soda manufacturing company. You are tasked with tracking your products being used in movies. You have a million hours of video that need to be analyzed and tagged. You have built a TensorFlow deep learning model and have used some custom TensorFlow Operations written in C++. You have tested the model locally on a Vertex AI Workbench instance and now want to train it on a larger dataset. What hardware options are available for training?",
            "options": [
            "A. Use TPU v4 in the default setting because it involves using very large matrix operations.",
            "B. Customize the TPU v4 size to match with the video and recompile the custom TensorFlow Operations for TPU.",
            "C. You cannot use GPU or TPU because neither supports custom operations.",
            "D. Use GPU instances because TPUs do not support custom operations."
            ],
            "answer": ["D. Use GPU instances because TPUs do not support custom operations."]
        },
        {
            "question": "You are a data scientist in a large retail chain and you’re trying to forecast the sales of each product in each store. For each store you have information like zip code (high cardinality) and hundreds of other features which needs to be bucketized and also scaled. You want to simplify the pipeline for this. What will be your approach?",
            "options": [
            "A. Use Dataflow to transform the data (zip code and other features) and then Vertex AI AutoML forecasting to train the model.",
            "B. Use Dataflow to transform the data (zip code and other features), store in Vertex Feature Store, and then use Vertex AI AutoML forecasting to train the model.",
            "C. Use Dataflow to transform the data (zip code and other features), store it in Vertex Feature Store, and then use BigQuery ML to train an ARIMA_Plus model.",
            "D. Use BigQuery Transform to apply hashing transform to Zip code, and Bucketizing transforms and then user BigQuery ML to train an ARIMA_Plus model."
            ],
            "answer": ["D. Use BigQuery Transform to apply hashing transform to Zip code, and Bucketizing transforms and then user BigQuery ML to train an ARIMA_Plus model."]
        }
    ]
    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='collapsed')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")


page_names_to_funcs = {
    "Sample Questions": sample,
    "Assessment": assessment,
    "Chapter 1: Framing ML Problems": chap1,
    "Chapter 2: Exploring Data and Building Data Pipelines": chap2,
    "Chapter 3: Feature Enginnering": chap3,
    "Chapter 4: Choosing the Right ML Infrastructure": chap4,
    "Chapter 5: Architecting ML Solutions": chap5,
    "Chapter 6: Building Secure ML Pipelines": chap6,
    "Chapter 7: Model Building": chap7,
    "Chapter 8: Model Training and Hyperparameter Tuning": chap8,
    "Chapter 9: Model Explainability on Vertex AI": chap9,
    "Chapter 10: Scaling Models in Production": chap10,
    "Chapter 11: Designing ML Training Pipelines": chap11,
    "Chapter 12: Model Monitoring, Tracking, and Auditing Metadata": chap12,
    'Chapter 13: Maintaining ML Solutions': chap13,
    'Chapter 14: BigQuery ML': chap14,
    'Bonus Exam 1': bonus1,
    'Bonus Exam 2': bonus2
}   

demo_name = st.sidebar.selectbox("Choose chapter", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
