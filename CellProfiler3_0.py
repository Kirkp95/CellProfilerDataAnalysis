import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
import pylab
import matplotlib.colors as colors

path='' # insert the path of your files
data_type='' # Nuclei/Cells/FAs
rootdir=path

def clean_data(df, data_type): # TODO: it clean the data taking only the columns needed from CellProfiler output
    # Selecting necessary columns based on data type
    if data_type == 'Nuclei':
        columns_to_keep = ['ImageNumber', 'ObjectNumber', 'AreaShape_Perimeter', 'AreaShape_Area',
                           'AreaShape_MinFeretDiameter', 'AreaShape_MaxFeretDiameter',
                           'AreaShape_Eccentricity', 'Location_Center_X', 'Location_Center_Y',
                           'AreaShape_Orientation', 'AreaShape_FormFactor', 'AreaShape_MajorAxisLength',
                           'AreaShape_Solidity']
    elif data_type == 'Cells':
        columns_to_keep = ['ImageNumber', 'ObjectNumber', 'AreaShape_Perimeter', 'AreaShape_Area',
                           'AreaShape_MinFeretDiameter', 'AreaShape_MaxFeretDiameter',
                           'AreaShape_Eccentricity', 'Location_Center_X', 'Location_Center_Y',
                           'AreaShape_Orientation', 'AreaShape_FormFactor', 'AreaShape_MajorAxisLength',
                           'AreaShape_Solidity']
    elif data_type == 'FAs':
        columns_to_keep = ['ImageNumber', 'ObjectNumber', 'AreaShape_Perimeter', 'AreaShape_Area',
                           'AreaShape_MinFeretDiameter', 'AreaShape_MaxFeretDiameter',
                           'AreaShape_Eccentricity', 'Location_Center_X', 'Location_Center_Y',
                           'AreaShape_Orientation', 'AreaShape_FormFactor', 'AreaShape_MajorAxisLength',
                           'AreaShape_Solidity']

    # Filtering the DataFrame
    cleaned_df = df[columns_to_keep]

    # Removing rows with 0 values in any column
    cleaned_df = cleaned_df[(cleaned_df != 0).all(axis=1)]

    return cleaned_df


def save_to_excel(df, output_path): # TODO: it saves the data as excel file
    df.to_excel(output_path, index=False)


def plot_data_distributions(df, data_type, output_path): # TODO: it plot as violin plot the data stored in the excel file
    # Plotting data distributions
    num_columns = len(df.columns)
    num_subplots = min(num_columns, 6)  # Maximum 6 columns in a subplot

    if num_columns > 5:
        num_rows = 2
        num_cols = math.ceil(num_columns / 2)
    else:
        num_rows = 1
        num_cols = num_columns

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    fig.suptitle(f'Morphological Analysis {data_type}', fontsize=16)

    for i, column in enumerate(df.columns):
        row = i // num_cols
        col = i % num_cols
        if num_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]





        sns.violinplot(data=df, y=column, ax=ax)
        sns.stripplot(data=df, y=column, color='black', ax=ax)
        ax.set_xlabel(data_type)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate title
    plt.savefig(output_path)
    plt.close()


def CPA(path): # TODO: create excels files starting from the CellProfiler output with the desired samples
    # Asking for input
    sample_folder = input('Sample folder?\n')
    data_type = input('Which kind of samples? (Nuclei/Cells/FAs)\n')

    # Changing directory
    os.chdir(path)
    path = os.getcwd()
    new_path = os.path.join(path, sample_folder)
    print(new_path)
    os.chdir(new_path)
    # for sub in os.listdir(new_path):
    #     print(sub)
    #     subnew_path = os.path.join(new_path,sub)
    #     os.chdir(subnew_path) #put subnewpath if there are more subfolders
        # Opening CSV files 

    if data_type == 'Nuclei':
        df_nuclei = pd.read_csv('input filename')
        df_cleaned = clean_data(df_nuclei, 'Nuclei')
        output_filename = f'GenMorpho_Nuclei_{sample_folder}.xlsx'
    elif data_type == 'Cells':
        df_cells = pd.read_csv('input filename')
        df_cleaned = clean_data(df_cells, 'Cells')
        output_filename = f'GenMorpho_Cells_{sample_folder}.xlsx'
    elif data_type == 'FAs':
        df_fas = pd.read_csv('input filename')
        df_cleaned = clean_data(df_fas, 'FAs')
        output_filename = f'GenMorpho_FAs_{sample_folder}.xlsx'
    else:
        print("Invalid data type entered. Please choose from 'Nuclei', 'Cells', or 'FAs'.")
        return


    # Saving cleaned data to Excel file
    save_to_excel(df_cleaned, output_filename)

    # Plotting data distributions
    plot_data_distributions(df_cleaned, data_type, f'{data_type}_{output_filename}_graph.png')

    # Q-Q plot
    measurements = df_cleaned['AreaShape_Area']
    stats.probplot(measurements, dist="norm", plot=pylab)
    pylab.show()

CPA(rootdir)

def concatenate_excel_files(folder_path, output_filename): # TODO: It combines the excel output if you have multiple folders with several samples
    """
    Concatenates all Excel files in the specified folder_path into a single Excel file.
    Saves the concatenated data into output_filename.

    Args:
    - folder_path (str): Path to the folder containing Excel files to concatenate.
    - output_filename (str): Name of the output Excel file.
    """
    dfs = []

    # Iterate over all files in the folder_path
    for folder in os.listdir(folder_path):
        subpath=os.path.join(folder_path, folder)
        for file in os.listdir(subpath):
            if file.endswith('.xlsx')and 'GenMorpho' in file:  # Consider only Excel files
                file_path = os.path.join(subpath, file)
                df = pd.read_excel(file_path)
                dfs.append(df)

        # Concatenate all dataframes
        concatenated_df = pd.concat(dfs, ignore_index=True)

        # Save concatenated dataframe to Excel file
        output_file_path = os.path.join(folder_path, output_filename)
        concatenated_df.to_excel(output_file_path, index=False)
        


def CompareCPA(data_type, dictData):

    dictlen = len(dictData)

    for column in dictData.values():
        f, axs = plt.subplots(1, dictlen,
                              figsize=(26, 13),
                              squeeze=False,
                              sharey=True,
                              sharex=True)

        plt.autoscale(enable=True, axis='y')

        sns.set(font_scale=2)

        if column.columns.get_loc(column) > 3:
            samplenum = 0

            for key, df in dictData.items():
                print(df)

                Nucleig = sns.violinplot(data=df,
                                      y=column,
                                      color='orange',
                                      ax=axs[0, samplenum])
                Nucleig.set(xlabel=key)

                Nucleig = sns.stripplot(data=df,
                                        y=column,
                                        color='black',
                                        ax=axs[0, samplenum])
                Nucleig.set(xlabel=key)

                samplenum = samplenum + 1

            cname = str(column)

            dfname = 'Comparison_' + cname

            f.suptitle(f'Morphological {data_type} Analysis ' + dfname)

            name = dfname + '_graph.png'

            plt.savefig(name)
        
# CompareCPA(rootdir,)
     
def FDA(path, folders, scale, data_type=None): # TODO: it scales and divide the data per parameter in a new excel file
    os.chdir(path)
    path = os.getcwd()
    sampleslist = folders.split(',')
    print(sampleslist)
    
    sampleDict = {}
    
    for sample in sampleslist:
        if data_type == 'Nuclei':
            nuclei_filename = f'GenMorpho_Nuclei_{sample}.xlsx'
            nuclei_data = pd.read_excel(nuclei_filename)
            data = nuclei_data
        elif data_type == 'Cells':
            cells_filename = f'GenMorpho_Cells_{sample}.xlsx'
            cells_data = pd.read_excel(cells_filename)
            data = cells_data
        elif data_type == 'FAs':
            fas_filename = f'GenMorpho_FAs_{sample}.xlsx'
            fas_data = pd.read_excel(fas_filename)
            data = fas_data
        else:
            continue
        
        columns_to_scale = []
        columns_to_notscale = []
        
        for column in data:
            wordscheck = ['Eccentricity', 'Location', 'Orientation', 'Solidity', 'FormFactor', 'Number']
            if any(word in column for word in wordscheck):
                print('Not scaling:', column)
                columns_to_notscale.append(column)
            elif 'AreaShape_Area' in column:
                print('Scaling squared:', column)
                data[column] = data[column] * scale**2
            else:
                print('Scaling:', column)
                columns_to_scale.append(column)
        
        scaled_columns = data[columns_to_scale] * scale
        data[columns_to_scale] = scaled_columns
        
        notscaled_columns = data[columns_to_notscale]
        data[columns_to_notscale] = notscaled_columns
                
        sampleDict[sample] = data
    
    os.chdir(path)
    os.makedirs('ComparisonDFs', exist_ok=True)
    finapath = os.path.join(path, 'ComparisonDFs')
    os.chdir(finapath)
    
    for column in data.columns:
        column_data = {sample: sampleDict[sample][column] for sample in sampleslist}
        df = pd.DataFrame(column_data)
        df.to_excel(f'Column_{column}_{data_type}.xlsx', index=False)
               


def violin_plotting(directory): # TODO: it return violin plots 
     # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            # Read the Excel file into a DataFrame
            filepath = os.path.join(directory, filename)
            df = pd.read_excel(filepath)

            # Create a single violin plot for all columns
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=df, inner='point')
            plt.title(f'Violin Plot for {filename}')
            plt.xlabel('Columns')
            plt.ylabel('Values')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plot_filename = f'{os.path.splitext(filename)[0]}_violin_plot.png'
            plt.savefig(os.path.join(directory, plot_filename))
            plt.close()
               
# violin_plotting(rootdir)

def FFHM(rootdir,keyword): # TODO: it compute the mean for each parameter and combine in a single excel file
    os.chdir(rootdir)

    dffinal=pd.DataFrame()
    
    sample_names = []
    
    for filename in os.listdir(rootdir):
        if filename.find(keyword)!=-1 and filename.find('Nuclei.xlsx')!=-1:
            print (filename)
            
            wordscheck = ['Location', 'Orientation', 'Number']
            if any(word in filename for word in wordscheck):
                print('Not taking into account:', filename)
                
            else:
                print('calculating the mean:',filename)
                df = pd.read_excel(filename)
                filenamewoext = os.path.splitext(filename)[0]
                mean_list = df.mean().tolist()
                dffinal[filenamewoext] = mean_list
                # Extract sample names from column headers
                sample_names = df.columns.tolist() if not sample_names else sample_names

    # Set sample names as index
    dffinal.index = sample_names
                # print(dffinal)
    dffinal.to_excel('MEANSNucleiFig1.xlsx')
         
# FFHM(rootdir,'Column')

def NormMean(rootdir): # TODO: method to normalize different samples

    #navigate and opening the excel files containing the mean values
    os.chdir(rootdir)
    df=pd.read_excel('MEANSNucleiFig3.xlsx')
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    #iterate over the df and choosing the CTRL value to normalize on

    for column  in df:

        if column != 'Samples' :

            standardvalue= df.loc[1, column] #change the number depending on the raw number of the sample to normalize on
            print (standardvalue)

            maxdifference = 0
            for col, values in df.items():

                if col == column:

                    for value in values:
                        print(value)
                        difference = abs(value - standardvalue)

                        if difference > maxdifference:
                            maxdifference = difference

            df[column] = (df[column] - standardvalue) / maxdifference
            print (df)


    else:
            pass

    # df.to_excel('MEANSNucleiScaledFig3.xlsx')
    
NormMean(rootdir)

def HeatMap(rootdir,file): # TODO: it returns a heatmap normlized on a desired sample 

    os.chdir(rootdir)
    # Read the Excel file
    df1 = pd.read_excel(file)

    # Exclude the first column from the dataframe
    df1 = df1[df1['Samples'] != "control"] #put the sample you are normalizing on
    data = df1.iloc[:, 2:]
    # plt.style.use("")

    print("Our dataset is : ", data)

    # Define custom color map in case of multiple objects
    colors_list = [(0, "Blue"), (0.5, "White"), (1, "Red")]
    custom_cmap = colors.LinearSegmentedColormap.from_list("", colors_list)
    print(data.T)
    # Create the heatmap
    plt.figure(figsize=(75, 75))
    heat_map = sns.heatmap(data.T, cmap=custom_cmap, linewidth=20, xticklabels=df1['Samples'], yticklabels=data.columns,
                           annot=False, annot_kws={'size': 30}, cbar_kws={"orientation": "horizontal","shrink": 0.3},square=True)

    # adjust spacing around cells
    plt.tight_layout(pad=3)
    
    plt.subplots_adjust(left=0.2,right=0.9)
    cbar = heat_map.collections[0].colorbar
    cbar.ax.tick_params(labelsize=50)

    # Set font sizes for tick labels and title
    for label in heat_map.xaxis.get_ticklabels():
        label.set_size(75)
    for label in heat_map.yaxis.get_ticklabels():
        label.set_size(90)
    # heat_map.set_title("HeatMap of morphometric parameters", fontsize=60)

    # for label in heat_map.yaxis.get_ticklabels():
    #     if 'Nuclei' in label.get_text():
    #         label.set_color('blue')
    #     elif 'Cell' in label.get_text():
    #         label.set_color('green')
    #     elif 'FAs' in label.get_text():
    #         label.set_color('red')
    #     else:
    #         label.set_color('black')

        # label.set_size(75)
    
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    # Rotate the x-axis tick labels
    plt.xticks(rotation=-45)

    # Display the plot and save as a PNG image
    # plt.show()
    plt.savefig('outputfilename')
    plt.savefig('outputfilename')
    plt.close()
    
# HeatMap(rootdir,'MEANSNucleiScaledFig3.xlsx')


def copy_rescaled_columns(source_file_path, destination_file_path, column_name):
    try:
        # Read the source and destination Excel files
        source_df = pd.read_excel(source_file_path)
        destination_df = pd.read_excel(destination_file_path)

        # Check if the column exists in the destination file
        if column_name not in destination_df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the destination file.")

        # Copy the specified column from the destination file to the source file
        source_df[column_name] = destination_df[column_name]

        # Write the modified source DataFrame back to the original source file
        source_df.to_excel(source_file_path, index=False)
        print(f"Column '{column_name}' copied successfully from '{destination_file_path}' to '{source_file_path}'.")
    
    except FileNotFoundError:
        print("File not found. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
