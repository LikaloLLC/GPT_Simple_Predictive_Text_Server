export GOOGLE_APPLICATION_CREDENTIALS=/Users/ptrv/PycharmProjects/docsie-firebase-analytics/analytics.json
functions-framework --target=docsie_analyze --port 8888
gcloud config set project docsie-analytics
gcloud functions deploy docsie_convert --trigger-http --runtime python37 --entry-point docsie_convert --allow-unauthenticated
