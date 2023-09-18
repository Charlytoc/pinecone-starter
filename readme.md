# Pinecone made easy

This project lets you use you own data to store it in Pinecone using just Pinecone and Langchain, you can feel free to research more in your own and use any piece of code found here.

To start this project is easy as just clone the repository, **you need to have Docker and Devcontainers installed.**

## Git clone the repo
```
git clone https://github.com/Charlytoc/pinecone-starter.git
```
## Open the repository directory
```
cd pinecone-starter
```
## Open the command palette in VSCode with F1

## Search for Devcontainer and press **Reopen in container**

## Copy the .env.example file in a .env file and add your keys
```
cp .env.example .env
```

## Run
```
python main.py --load_data
```
This will run the main.py script to load all the data from the docs directory to Pinecone using the default index name. You can specify the index name using the flag --index_name=your-index-name. You can also run without the --load_data flag to avoid reloading the data.

Then the command line will prompt you to ask something about the data. By default I already added a text about a movie called **Sound of Freedom**, you can ask anything about it!

