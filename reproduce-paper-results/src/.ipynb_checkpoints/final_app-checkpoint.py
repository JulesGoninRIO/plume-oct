from tkinter import filedialog, messagebox
import tkinter as tk
import os
import sys
from PIL import Image,ImageTk
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
project_path = "T:/Studies/OCTGWAS/code/Alicia/oct_quality"
assert os.path.exists(project_path)
sys.path.insert(1, project_path)
from src  import image_load_app, oct_plot


positions = {'geometry':["800x600","1420x720"],
             'vertical':[(700,150),(1330,200)],
             'horizontal':[(250,540),(880,570)],
             'save':[(50,200),(1350,600)],
             'misalignment':[(30,50),(1300,650)],
             'title':[(200,5),(825,30)],
             'name':[5,355],
             'detected':[800,1100],
             'pady':[20,5]}


class OCT_viewer(tk.Tk):
    
    
    def __init__(self): # set up main window, tkinter widgets and menu
    
        super().__init__()
        self.modality = ''
        self.oct_name = None
        self.resizable(width=False,height=False)
        self.title('OCT Annotation Assistant')
        menubar = tk.Menu(self) # menu architecture
        self.menu = tk.Menu(menubar,tearoff=0)
        self.menu.add_command(label='Select folder',command=lambda:self.start())
        self.menu.add_command(label='Load annotation',command=lambda:self.load(''))
        self.menu.add_command(label='Compare annotations',command=lambda:self.compare_annotations())
        menubar.add_cascade(label="Menu",menu=self.menu)
        self.config(menu=menubar)
        self.image_path = "T:/Studies/OCTGWAS/code/Alicia/oct_quality/images/oct.jpg"
        image = ImageTk.PhotoImage(Image.open(self.image_path))
        self.label = tk.Label(self,image=image)
        self.label.grid()
        self.mainloop()
    
    
    def start(self):
        
        self.compare = 0
        if not self.modality:
            self.folder_path = filedialog.askdirectory(title='Select directory')+'/' # recording folder path
            if self.folder_path == '/':
                return
            modality = self.folder_path[-4:-1]
            self.index = -self.folder_path[:-1][::-1].index("/")-1
            if os.path.exists(self.folder_path[:self.index] + "annotation_" + modality + ".csv"):
                if messagebox.askyesno(title="Info", message="A csv file with self.format " + modality +
                                       " already exists. Do you want to load it ?"):
                    self.load(self.folder_path[:self.index] + "annotation_" + modality + ".csv")
                else:
                    self.destroy()
                    OCT_viewer()
        else:
            self.folder_path = self.csv_path[:-self.csv_path[:-1][::-1].index("/")-1]+"OCTs_"+self.modality+"/"
        self.place_objects()
        
    
    def load(self,path):
        
        if not path:
            self.csv_path = filedialog.askopenfilename(title='Select csv',filetypes=[("CSV files", "*.csv")])
        else:
            self.csv_path = path
        df = pd.read_csv(self.csv_path)
        self.modality = self.csv_path[-7:-4]
        self.bad = [[tk.IntVar(value=0) for j in range(int(self.modality))] for i in range(len(df))]
        df["frames"] = df["frames"].apply(literal_eval)
        data = [[uuid,frames] for uuid, frames in zip(df['uuid'], df['frames'])]
        for i in range(len(data)):
            for j in data[i][1]:
                self.bad[i][j-1].set(1)
        self.start()
        
    
    def place_objects(self):
        
        self.oct,self.fundus,self.frame = tk.IntVar(value=1),tk.IntVar(value=1),tk.IntVar(value=1)
        self.button,self.oct_title = tk.Checkbutton(),tk.Text()
        self.root_folder = ''
        for name in self.folder_path[:-1].split('/'):
            if not name.startswith('OCT_annotation'):
                self.root_folder += name+'/'
        self.oct_names = os.listdir(self.root_folder)
        nb_frames = len([file for file in os.listdir(self.root_folder+self.oct_names[0]) if file.endswith(".jpg")])
        if not self.modality or self.compare:
            self.bad = [[tk.IntVar(value=0) for j in range(nb_frames)] for i in range(len(self.oct_names))]
        (x,y) = positions['save'][self.compare]
        tk.Button(text='save',command=self.save).place(x=x,y=y)
        (x,y) = positions['horizontal'][self.compare]
        horizontal_scale = tk.Scale(self,variable=self.frame,from_=1,to=nb_frames,orient='horizontal',
                                    length=300,bg='white',command=lambda x:self.change_frame())
        horizontal_scale.place(x=x,y=y) 
        horizontal_scale_text = tk.Text(self,width=len("Change B-scan"),height=1,relief=tk.FLAT)
        horizontal_scale_text.insert(tk.INSERT,"Change B-scan")
        horizontal_scale_text.place(x=x+100,y=y-20)
        (x,y) = positions['vertical'][self.compare]
        vertical_scale = tk.Scale(self,variable=self.oct,from_=1,to=len(self.oct_names),orient='vertical',
                                         length=200,bg='white',command=lambda x:self.change_frame())
        vertical_scale.place(x=x,y=y) 
        vertical_scale_text = tk.Text(self,width=len("Change OCT"),height=1,relief=tk.FLAT)
        vertical_scale_text.insert(tk.INSERT,"Change OCT")
        vertical_scale_text.place(x=x-30,y=y-30)
        self.geometry(positions['geometry'][self.compare])
        self.bind("<Left>",lambda e:horizontal_scale.set(horizontal_scale.get()-1)) 
        self.bind("<Right>",lambda e:horizontal_scale.set(horizontal_scale.get()+1))
        self.bind("<Up>",lambda e:vertical_scale.set(vertical_scale.get()-1)) 
        self.bind("<Down>",lambda e:vertical_scale.set(vertical_scale.get()+1))
        
        self.change_frame()
    
    
    def change_frame(self):
        
        i = str(self.frame.get())
        oct_name = self.oct_names[self.oct.get()-1]
        self.oct_title.grid_forget()
        self.oct_title = tk.Text(self,width=len(oct_name),height=1,relief=tk.FLAT)
        self.oct_title.insert(tk.INSERT,oct_name)
        (x,y) = positions['title'][self.compare]
        self.oct_title.place(x=x,y=y)
        
        self.button.grid_forget()
        self.button = tk.Checkbutton(text='Misalignment',variable=self.bad[self.oct.get()-1][self.frame.get()-1])
        (x,y) = positions['misalignment'][self.compare]
        self.button.place(x=x,y=y)
        
        image = Image.open(self.root_folder+oct_name+'/'+i+'.jpg')
        image = ImageTk.PhotoImage(image)
        self.label.configure(image=image)
        self.label.image = image
        
        if not self.compare:
            self.label.place(height=500, width=800)
        else:
            if oct_name != self.oct_name:
                for fundus in os.listdir(self.fundus_path[0]):
                    if fundus.find(oct_name) != -1:
                        self.oct_name = oct_name
                        for k in range(2):
                            image_left = Image.open(self.fundus_path[k]+fundus)
                            image_left = ImageTk.PhotoImage(image_left.resize((720,720)).crop((0,200,720,520)))
                            self.label_left[k].configure(image=image_left)
                            self.label_left[k].image = image_left
            for k in range(2):
                if int(i) in self.annotations[k][self.oct_names[self.oct.get()-1][-36:]]:
                    self.display_text[k].set(self.name[k]+" annotated peak")
                    self.display[k].configure(bg='#fff', fg='#f00')
                else:
                    self.display_text[k].set("")
    
    
    def save(self):
        
        misalignments = {}
        modality = self.folder_path[-4:-1]
        for i in range(len(self.oct_names)):
            misalignments[self.oct_names[i][-36:]] = []
            for j in range(int(modality)):
                if self.bad[i][j].get():
                    misalignments[self.oct_names[i][-36:]].append(j+1)
        index = -self.folder_path[:-1][::-1].index("/")-1
        save_file = 'annotation_'+modality+'.csv'
        if self.compare:
            save_file = self.name[0] + "_" + self.name[1] + "_" + save_file
        with open(self.folder_path[:index]+'/'+save_file, 'w', newline='') as csvfile:
           csvwriter = csv.writer(csvfile, delimiter=',')
           csvwriter.writerow(["uuid", "modality", "frames"])
           for key, value in misalignments.items():
               csvwriter.writerow([key, modality, value])
                    
               
    def read_annotation(self,i):
        
        csv_path = filedialog.askopenfilename(title='Select directory',filetypes=[("CSV files", "*.csv")])
        self.modality = csv_path[-7:-4]
        df_annotations = pd.read_csv(csv_path)
        for index, row in df_annotations.iterrows():
            frame_list_str = row["frames"]
            frame_list = literal_eval(frame_list_str)
            self.annotations[i][row["uuid"]] = frame_list
        self.index = -csv_path[:-1][::-1].index("/")-1
        output_path = csv_path[:self.index]
        self.fundus_path[i] = output_path + "fundus_annotations/"
        self.name[i] = os.path.basename(os.path.dirname(csv_path)).split("_")[2]
        self.folder_path = output_path + "OCTs_" + self.modality + '/'
        
        if not os.path.exists(self.fundus_path[i][:-1]):
            os.mkdir(self.fundus_path[i][:-1])
            
        df = pd.read_csv(output_path + "POC_score.csv")
        for row in df.iterrows():
            if row[1]["dataset_uuid"] in self.annotations[i].keys():
                compound_metric = np.array(row[1][len(self.headers):].to_list())
                compound_metric = compound_metric[compound_metric != -9999.0]
                figname = "_PID" + str(row[1]['patient_id']) + "_UUID" + str(row[1]['dataset_uuid'])
                if not os.path.exists(self.fundus_path[i] + figname + ".png"):
                    volume = np.array(image_load_app.load_images_from_folder(row[1]['folder_path']))
                    fundus = oct_plot.compute_fundus(volume)
                    oct_plot.fundus_along_POC(fundus, compound_metric, 'compound', x_max=max(np.max(compound_metric)+10,50), 
                                              peak_list = self.annotations[i][row[1]["dataset_uuid"]], show_curve = False)
                    plt.savefig(self.fundus_path[i] + figname + '.png')
                    plt.close()
                              
        self.label_left[i] = tk.Label(self.left_frame, image=ImageTk.PhotoImage(Image.open(self.image_path)))
        self.label_left[i].grid(row=1+i, column=0, padx=5, pady=positions['pady'][i])
        
        detected_frame = tk.Frame(self, width=100, height=100, bg='white')
        detected_frame.place(x=positions['detected'][i], y=650, relx=0.01, rely=0.01)
        self.display_text[i] = tk.StringVar()
        self.display[i] = tk.Label(detected_frame, textvariable=self.display_text[i])
        self.display[i].grid(row=0, column=0, padx=10, pady=5)
        
        name_text = tk.Text(self,width=len(self.name[i]),height=1,relief=tk.FLAT)
        name_text.insert(tk.INSERT,self.name[i])
        name_text.place(x=330,y=positions['name'][i])
        name_text.configure(bg='#fff', fg='#f00')


    def compare_annotations(self):
        
        self.config(bg="skyblue")
        self.compare = 1
        self.headers = ['patient_id', 'date', 'study_uuid', 'dataset_uuid', 'modality', 'laterality', 'folder_path']
        self.annotations = [{},{}]
        self.fundus_path,self.name,self.label_left,self.display,self.display_text = ['',''],['',''],['',''],['',''],['','']
        
        self.left_frame = tk.Frame(self, width=300, height= 100, bg='grey')
        self.left_frame.grid(row=0, column=0, padx=10, pady=5) 
        self.right_upper_frame = tk.Frame(self, width=300, height=100, bg='grey')
        self.right_upper_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.label = tk.Label(self.right_upper_frame, image=ImageTk.PhotoImage(Image.open(self.image_path)))
        self.label.grid(row=0, column=0, padx=5, pady=5)

        self.read_annotation(0),self.read_annotation(1)
        self.place_objects()


if __name__ == '__main__':
    OCT_viewer()