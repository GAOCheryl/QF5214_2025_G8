# QF5214_Group 8 Team_1 README    

## 1 Stock_data/
- Contains files related to stock data(open, close, etc.).
  
## 2 X_data/
- **2.1 The X_1 and X_3 directories**
  - contain the first version of the script and its execution method. Each time, we manually changed the names of four companies in the script, pulled the code from GitHub, and ran it on Alibaba ECS. This method was executed twice, and the resulting text data can be found in the respective folders:
  - X_1 contains data for the companies: "ADBE", "AMD", "ABNB", "GOOGL"
  - X_3 contains data for the companies: "ADI", "ANSS", "AAPL", "AMAT"
- **2.2 The getX folder**
  - Contains the second version of the script and its execution method.
  - **Script Modifications:**
    - The script has been modified to include `input()` prompts for the company names and the output filename.
    - Each time the code runs, a user interaction dialog appears for entering the desired company list and file name.
    - This modification is designed to allow the simultaneous execution of scripts with different config.ini files, different company name lists, and different output file names.
  - **Execution Method:**
    - The script runs within a Docker container on Alibaba Cloud ECS.
    - The Docker image is built as described in the Dockerfile.
    - The Docker container is executed using the following shell command:
    ```sh
    docker run -it \
     --name getx_2 \        # Name the container accordingly.
     -v /root/QF5214_G8_Team1_Config/2/config.ini:/getX/config.ini \        # Mount the host's config.ini file into the container at /getX/config.ini.
     -v /root/QF5214_G8_Team1_Config/2:/output \        # Mount the host directory to /output in the container for saving output files.
     crpi-hdkevh0503yezvt4.ap-southeast-1.personal.cr.aliyuncs.com/qf5214_g8t1_getx/qf5214_g8:latest \        # Specify the Docker image to use from the Alibaba Cloud Container Registry.
     /bin/sh -c "python get_X_text.py && cp /getX/*.csv /output/"        # Run a shell command that executes the Python script and then copies any CSV files from /getX to /output.
    ```
- **2.3 Tweets**
  - Used to temporarily store the script's output (considering that the CSV files are relatively small, we decided to output everything as CSV files first. Once data for all companies has been scraped, the files will be collectively written into the PostgreSQL database).

## 3 sql transformation.ipynb
  Demonstrating how to perform SQL-related data transformations.(from csv to PostGreSQL)

## 4 README.md
  The current document, providing an overview of the project, its structure, and usage instructions.
