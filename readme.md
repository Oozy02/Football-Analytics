# Predictive Insights for Financial Recovery in Club Football

This project aims to leverage data analysis and machine learning to uncover predictive insights for financial recovery and strategic optimization in club football. Key questions explored include identifying undervalued players, evaluating financially beneficial matches, analyzing impactful player attributes, and determining optimal player combinations.

---

## Project Questions and Authors

The project focuses on the following questions, which are **boldly highlighted** in the code for easy reference:

1. **Which players are undervalued in the transfer market and could potentially yield higher returns in the future?**  
   *Author: Sahil Kakad (50607550)*  

2. **Which competitions or matches are the most financially beneficial for the club?**  
   *Author: Manas Kalangan (50608803)*  

3. **Which player combinations and lineups perform the best in terms of match success and financial returns?**  
   *Author: Sahil Kakad (50607550)*  

4. **How do different player attributes and performance metrics impact transfer fees?**  
   *Author: Manas Kalangan (50608803)*  

---

## Project Structure

### Root Directory
**`50608803_50607550_phase_3`**  
The main directory contains the project report and subfolders for the application and experimental analysis:
- `project_report.pdf`: Detailed report of the entire project.

### Subfolders

#### `app/`
Contains all application-related code:
- **`archives/`**: Dataset CSV files.
- **`database/`**: 
  - `database.py`: Handles database operations.
- **`models/`**: All machine learning model Python files.
- **`app.py`**: Main Streamlit application file.
- **`requirements.txt`**: Lists Python dependencies for the app.
- **`setup_and_run.sh`**: Shell script to set up and run the app.

#### `exp/`
Contains all experimental and analysis code:
- **`archives/`**: Dataset CSV files.
- **`football_analysis.ipynb`**: Final Jupyter Notebook with questions and hypotheses highlighted.
- **`requirements.txt`**: Lists Python dependencies for the experiments.

---

## Experiment Code and Reference

The Jupyter Notebook `football_analysis.ipynb` contains the experimental code associated with each project question. The hypotheses and related sections are **boldly highlighted** alongside the authors for easy reference.



## Instructions to Run the Code (Jupyter Notebook)

1. Extract the contents of the ZIP file to a directory of your choice. 
2. Open the `football_analysis.ipynb` notebook in Jupyter or any compatible Python environment (e.g., VS Code with Jupyter extension). 
3. Execute the notebook cells sequentially to view the code, analysis, and visualizations.

---
## Streamlit Application Stack 

- Application (Frontend + Backend ): streamlit 
- Database : Sqlite 

## Build and Run Instructions (App)

1. Navigate to the `app` directory:
   ```bash
   cd app
   chmod +x setup_and_run.sh  # If linux based 
   ./setup_and_run.sh


## Requirements

To run the code, you will need the following:

- Python 3.10 and above
- Jupyter Notebook or any compatible IDE (e.g., VS Code)
- Libraries: Mentioned in the requirements.txt 

## Important Notes 

- Train Model Page Password : 'admin_password'
- Command to terminate the server: control + c 
- Command to run the app when inside the virtual environment and in the app directory : streamlit run app.py 
- The video has been created on Windows environment using the git bash. There are some issues at time to run the .sh file. kindly, refer the full video.


## Purpose and Contribution

This analysis was created by **Sahil Kakad** and **Manas Kalangan**. The goal is to provide insights into the financial and performance metrics of football clubs, including player valuation, competition profitability, and team strategies. The findings from this project are intended to help football clubs optimize their financial strategies and improve on-field success.

## License

This project is released under the MIT License.

---
If you have any questions or feedback, feel free to reach out!
