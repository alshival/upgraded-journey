# upgraded-journey
Data Science & Analytics Tips/Tricks

# Large Language Models: An Application in Data Processing

Large Language Models (LLM) are more than just text generators. It was only in the past few years that we began to see larger LLMs appear that are capable of translating between languages. Here, I would like to discuss a few very simple applications in data processing. 

Unfortunately, there are not many open-source solutions out there that make processing data using LLMs, meaning that processing data using LLMs is still very much in its early stages.

Let's start with setting up the code.

## Preamble

First, we import the required packages for this document, and we will be using [openAi](https://openai.com/)'s davinci-003 model. You will need to sign up and get an api key.

Set your gpt_api key as an environmental variable called gtp_api_key or just gpt_api. See instructions on how to do so. On current linux archictecutres, you can add this line to your .bashrc file.

```
export gtp_api_key="<YOUR KEY>"
```


```python
import regex
import csv
import pandas as pd
import openai
import json
import sqlite3
import time
import os

openai.api_key = os.environ['gpt_api']
```

### Code to process text using openAi

This next function is something I put together quickly. It is a wrapper to speed up text processing with davinci.


```python
def process_text(command,data):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"""Q: {command}:"{data}"\nA:""",
        temperature=.5,
        max_tokens=120,
        top_p=.5,
        frequency_penalty=.5,
        presence_penalty=0,
        stop=["\n\n"])
    return response.choices[0].to_dict()['text']
```

Here is an example:


```python
process_text("tell me an interesting fact about this number","1279")
```




    ' 1279 is the smallest number that can be written as the sum of two cubes in two different ways: 1279 = 13^3 + 10^3 = 9^3 + 18^3.'





Srinivasa Ramanujan gave us that one already, davinci. I was hoping for something new, but thank you. 

How about this one: In 1279, the discovery of Mary Magdalene in the crypt of the church of Saint-Maximin, southeastern France.

Ah. Ramanujan worked with 1729. You are right.
The Ai fell for my trap.
I kind of knew they would. But it was a good way of bringing up Mary.

### Code to process text in a dataset using openAi
Anyways, this next function allows us to run the `process_text` function iteratively over a dataset. We add a `time.sleep` to pause the script in order to avoid hitting openAi's rate limits.


```python
def process_data(command,data,input_var='text',output_var='response'):
    outputs = []
    for i in range(len(data)):
        text = data.iloc[i][input_var]
        output = process_text(command,text)
        
        outputs = outputs + [output]
        
        time.sleep(2)
        
    return pd.concat([data,pd.DataFrame({output_var:outputs})],axis=1)
```

This last function is extremely useful. Let's look at a few examples.

## Example 1: Categorizing Data
Suppose we have the following dataframe and want to classify the items in the list as either `animal`, `flower`, `planet`, or `other`.


```python
data = pd.DataFrame({'id':[1,2,3,4,5,6,7,8,9],
                     'item':['cat','pen','whistle','orchid','Jupiter','car','dog','book','lotus']})
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>pen</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>whistle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>orchid</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Jupiter</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>car</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>book</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>lotus</td>
    </tr>
  </tbody>
</table>
</div>



We can do this easily with davinci's help!


```python
process_data('classify this as either animal, flower, planet, or other',data,input_var='item')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>item</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>cat</td>
      <td>Animal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>pen</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>whistle</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>orchid</td>
      <td>Flower</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Jupiter</td>
      <td>Planet</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>car</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>dog</td>
      <td>Animal</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>book</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>lotus</td>
      <td>Flower</td>
    </tr>
  </tbody>
</table>
</div>



# Example 2: Translating Text

Though openAi's davinci can translate text into a variety of languages, it does make mistakes which will require some manual intervention to fix, though when you have so many lines of text that need translating, a rough sketch of a translation can help speed up the process.

For this example, let's look at the first 8 lines of King Samuel's Gospel of Mary Magdalene (KSGM). You can obtain the .csv file via the [**super_bible**](https://github.com/alshival/super_bible) project.


```python
ksgm = pd.read_csv('super_bible_KSGM.csv').head(8)
ksgm
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>testament</th>
      <th>book</th>
      <th>title</th>
      <th>chapter</th>
      <th>verse</th>
      <th>text</th>
      <th>version</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>0</td>
      <td>[Pages 1 through 6 are missing]</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>1</td>
      <td>``Will matter be destroyed?''</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>2</td>
      <td>The Savior said:</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>3</td>
      <td>``Every form of nature, every creature, exists...</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>4</td>
      <td>Peter said to Him: ``As you have told us all a...</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>5</td>
      <td>The Savior answered:</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>6</td>
      <td>``There is no sin of the world. It is you who ...</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>7</td>
      <td>Because of this, the Lord comes into your mids...</td>
      <td>KSGM</td>
      <td>EN</td>
    </tr>
  </tbody>
</table>
</div>



Suppose we want to translate the text into French. We can do so easily by using the `process_data` function.


```python
process_data('translate this into french',ksgm)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>testament</th>
      <th>book</th>
      <th>title</th>
      <th>chapter</th>
      <th>verse</th>
      <th>text</th>
      <th>version</th>
      <th>language</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>0</td>
      <td>[Pages 1 through 6 are missing]</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>Les pages 1 à 6 manquent.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>1</td>
      <td>``Will matter be destroyed?''</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>``La matière sera-t-elle détruite ?''</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>2</td>
      <td>The Savior said:</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>Le Sauveur a dit :</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>3</td>
      <td>``Every form of nature, every creature, exists...</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>``Toute forme de la nature, chaque créature, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>4</td>
      <td>Peter said to Him: ``As you have told us all a...</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>Pierre lui a dit : « Comme vous nous avez tou...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>5</td>
      <td>The Savior answered:</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>Le Sauveur a répondu :</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>6</td>
      <td>``There is no sin of the world. It is you who ...</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>Il n'y a pas de péché dans le monde. C'est vo...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NT</td>
      <td>777</td>
      <td>Gospel of Mary Magdalene</td>
      <td>1</td>
      <td>7</td>
      <td>Because of this, the Lord comes into your mids...</td>
      <td>KSGM</td>
      <td>EN</td>
      <td>En raison de cela, le Seigneur vient au milie...</td>
    </tr>
  </tbody>
</table>
</div>



Note that I left out the `input_var` variable. This is because in the function definition for `process_data`, you will see that `input_var='text'`, which happens to be the name of the column that I wish to process.

# Example 3: Data Insight

The `process_text` function is more verstatile than you think. It returns the text response from openAi's algorithm.

You can use the `process_text` function on a pandas dataframe in order to gain insight.


```python
print(data)
print('\n')
process_text('classify the items in this dataset',data)
```

       id     item
    0   1      cat
    1   2      pen
    2   3  whistle
    3   4   orchid
    4   5  Jupiter
    5   6      car
    6   7      dog
    7   8     book
    8   9    lotus

    ' Animals: cat, dog\nStationery: pen\nInstruments: whistle\nPlants: orchid, lotus\nPlanets: Jupiter\nVehicles: car\nObjects: book'



We could have also done the same with the KSGM dataset.


```python
process_text('translate the field called text in this dataset into french', ksgm)
```




    ' le champ appelé texte dans ce jeu de données en français: "testament livre titre chapitre verset\n0        NT   777  Évangile de Marie Madeleine        1      0   \n1        NT   777  Évangile de Marie Madeleine        1      1   \n2        NT   777  Évangile de Marie Madeleine        1      2   \n3        NT   777  Évangile de Marie Madeleine        1      3   \n'



This method is quicker than the iterative process above, though given that the output of these models is subject to change, I do not feel confident building code dependent on these outputs. Developmentally, we are still at a crawling phase with these models.
