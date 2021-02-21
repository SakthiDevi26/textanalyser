from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from app import SentimentPredictor
#from app import Summarizer

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'summarize'

mysql = MySQL(app)

import os

sentiment_model_path=os.path.join(app.instance_path,'/tcs/flask/app/sentiment_model_weights')
print(sentiment_model_path)
sentiment = SentimentPredictor(model_weight_path=sentiment_model_path)

#summarizer_model_path=os.path.join(app.instance_path,'/tcs/flask/app/summarizer_model_weights')
#summarizer = Summarizer(model_weight_path=summarizer_model_path)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/summarize',methods = ['POST','GET'])
def summarize():
    if request.method=='POST':
        text=request.form['text']
        print(text)
        output = summarizer.predict(text)
        print('-------output-----------')
        print(output)
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO textsummarize(input_text,output_text) VALUES (%s, %s)', (text,output))
        mysql.connection.commit()
        cur.close()
        return render_template("summarize.html", text = output)
    elif request.method=='GET':
        return render_template("summarize.html")
    
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/analyze',methods = ['POST','GET'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        print(text)
        #output = 'positive'
        output = sentiment.predict(text)
        print(output)
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO textanalyze(input_text,output_text) VALUES (%s, %s)', (text, output))
        mysql.connection.commit()
        cur.close()
        return render_template("analyze.html", text=output)
    elif request.method == 'GET':
        return render_template("analyze.html")
if __name__=="__main__":
    app.run(debug=True)
