import flask
import pickle
import pandas as pd

with open(f'model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'model/ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open(f'model/dic_cat.pkl', 'rb') as f:
    dic_cat = pickle.load(f)
with open(f'model/lis_num.pkl', 'rb') as f:
    lis_num = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html', 
                                    lis_num = lis_num,
                                     dic_cat = dic_cat))
    
    if flask.request.method == 'POST':
        
        dic_num_val = {}
        for i in lis_num:
            dic_num_val[i] = flask.request.form[i]
        df_num = pd.DataFrame(dic_num_val, index=[0])
        
        dic_cat_val = {}

        for i in dic_cat.keys():
            dic_cat_val[i] = flask.request.form[i]
        df_cat = pd.DataFrame(ohe.transform(pd.DataFrame(dic_cat_val, index=[0])), index=[0])


        dfinput = pd.concat([df_cat, df_num], axis=1)
        
        prediction = model.predict_proba(dfinput.values)[0][1]
    
        return flask.render_template('main.html',
                                     prediction=round(prediction*100,2),
                                     lis_num = lis_num,
                                     dic_cat = dic_cat,
                                     )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)