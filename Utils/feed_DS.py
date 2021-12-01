import os
import pandas as pd
import shutil


orig_path = '/home/lizano/Documents/CSA-Loop/SNN/DS'
landing_path = '/home/lizano/Documents/CSA-Loop/CNN/DS/SNN_DS/Dump'

for v_dir in os.listdir(orig_path):
	v_path = orig_path+'/'+str(v_dir)
	if os.path.isdir(v_path):
		for sampling_dir in os.listdir(v_path):
			sampling_path = v_path+'/'+str(sampling_dir)
			if os.path.isdir(sampling_path):
				for traj_dir in os.listdir(sampling_path):
					traj_path = sampling_path+'/' + str(traj_dir)
					if os.path.isdir(traj_path):
						csv_name = v_dir+'-'+sampling_dir+'-'+traj_dir+'.csv'
						data = pd.read_csv(traj_path+'/'+csv_name)
						img_path = traj_path+'/plots'
						for index,rows in data.iterrows():
							img_name = '/'+v_dir+'-'+traj_dir+'-'+str(index)+'step-'+sampling_dir+'.png'
							dst = landing_path+'/'+str(int(rows['S_param']))
							shutil.copy(img_path+img_name, dst+img_name)