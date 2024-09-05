"""
A set of plotting functions
"""
import os, random, warnings, cv2, math
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import pandas as pd
import numpy as np
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from src.helpers.utils import get_slice, get_config
from src.modules.preprocessing import get_transformations


__all__ = ['counter', 'single_sample', 'numerical_features', 'categorical_features', 'confusion_matrix', 'training_values', 'image_transformations', 'results', 'gradcam']


def counter(data_folder, meta_folder):
	"""
	Plots data counters.
	Args:
		data_folder (str): the path of the folder containing images.
		meta_folder (str): the path of the folder containing csv files.
	Returns:
		None.
	"""
	mr_sessions = sorted([f.split('_')[0] for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))])
	n_subjects = len(list(set(mr_sessions)))
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
	bar_labels = ['MR Sessions (n.'+str(len(mr_sessions))+')', 'Nr. Subjects (n.'+str(n_subjects)+')']
	bars = ax1.bar(['MR Sessions', 'Nr. Subjects'], height=[len(mr_sessions), n_subjects], label=bar_labels, color=['#8fce00', '#ff8200'])
	for rect in bars:
		height = rect.get_height()
		ax1.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f} ({height/len(mr_sessions)*100:.2f}%)', ha='center', va='bottom', fontsize=14)
	ax1.set_title('Comparison nr. of SESSIONS and SUBJECTS', fontsize=20, fontweight='bold')
	ax1.tick_params(axis='both', which='major', labelsize=14)
	ax1.legend(fontsize=14)
	occur = np.unique(mr_sessions, return_counts=True)[1]
	values, counters = np.unique(occur, return_counts=True)
	bars = ax2.bar(['MR'+str(v) for v in values], height=counters, label=[str(i)+'-sessions' for i in values], color=['#8fce00', '#0092ff', '#ff8200', '#ff1100', '#ffcd34'])
	for rect in bars:
		height = rect.get_height()
		ax2.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f} ({height/n_subjects*100:.2f}%)', ha='center', va='bottom')
	ax2.set_title('SUBJECTS by NR. OF SESSIONS', fontsize=20, fontweight='bold')
	ax2.tick_params(axis='both', which='major', labelsize=14)
	ax2.legend(fontsize=14)
	df = pd.read_csv(os.path.join(meta_folder, 'data_num.csv'))
	data = {
		'oas4': [
			len(df[(df['oasis_id'] == 'OAS4') & (df['final_dx'] == 1.)]),
			len(df[(df['oasis_id'] == 'OAS4') & (df['final_dx'] != 1.)])
		],
		'oas3': [
			len(df[(df['oasis_id'] == 'OAS3') & (df['final_dx'] == 1.)]),
			len(df[(df['oasis_id'] == 'OAS3') & (df['final_dx'] != 1.)])
		]
	}
	bars = ax3.bar(['AD', 'Non-AD'], height=data['oas4'], label=['AD', 'Non-AD'], color=['#008DDA', '#008DDA'])
	for rect in bars:
		height = rect.get_height()
		ax3.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f} ({height/sum(data["oas4"])*100:.2f}%)', ha='center', va='bottom', fontsize=14)
	ax3.set_title('OASIS_4', fontsize=20, fontweight='bold')
	ax3.tick_params(axis='both', which='major', labelsize=14)
	ax3.legend(fontsize=14)
	bars = ax4.bar(['AD', 'Non-AD'], height=data['oas4'], label='OASIS_4', color=['#008DDA'], bottom=[0, 0])
	ax4.bar_label(bars, label_type='center', fontsize=14)
	bars = ax4.bar(['AD', 'Non-AD'], height=data['oas3'], label='OASIS_3', color=['#FF204E'], bottom=data['oas4'])
	ax4.bar_label(bars, label_type='center', fontsize=14)
	for k, rect in enumerate(bars):
		height = data['oas4'][k] + data['oas3'][k]
		ax4.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f} ({height/(sum(data["oas4"])+sum(data["oas3"]))*100:.2f}%)', ha='center', va='bottom', fontsize=14)
	ax4.set_title('FINAL DATASET', fontsize=20, fontweight='bold')
	ax4.tick_params(axis='both', which='major', labelsize=14)
	ax4.legend(fontsize=14)
	fig.tight_layout()
	plt.show()


def single_sample(folder):
	"""
	Plots sample data.
	Args:
		folder (str): the path of the folder containing images.
	Returns:
		None.
	"""
	samples = random.sample([i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))], 1)
	images = sorted([i for i in os.listdir(os.path.join(folder, samples[0])) if i != '.DS_Store'])
	brain_vol_t1 = nib.load(os.path.join(folder, samples[0], images[1]))
	brain_vol_t2 = nib.load(os.path.join(folder, samples[0], images[0]))
	fig, axs = plt.subplots(3, 2, figsize=(18, 14))
	for i, axl in enumerate(axs):
		if i == 0:
			axl[0].set_title('Scan mode '+images[0].split('_')[-1][:3])
			axl[1].set_title('Scan mode '+images[1].split('_')[-1][:3])
		axl[0].imshow(get_slice(brain_vol_t1, i, int(brain_vol_t1.get_fdata().shape[i] / 2)), cmap='gray')
		axl[1].imshow(get_slice(brain_vol_t2, i, int(brain_vol_t2.get_fdata().shape[i] / 2)), cmap='gray')
		axl[0].axis('off'), axl[1].axis('off')
	warnings.filterwarnings('ignore')
	plotting.plot_anat(brain_vol_t1, display_mode='x', title=images[0])
	plotting.plot_anat(brain_vol_t1, display_mode='y', title=images[0])
	plotting.plot_anat(brain_vol_t1, display_mode='z', title=images[0])
	fig.tight_layout()
	plt.show()


def numerical_features(folder):
	"""
	Plots numerical features.
	Args:
		folder (str): the path of the folder containing csv files.
	Returns:
		None.
	"""
	df = pd.read_csv(os.path.join(folder, 'data_desc.csv'))
	df1 = df[(df['weight'] != .0) & (df['height'] != .0)]
	df['bmi'] = round(df1['weight'] / (df1['height'] * df1['height']), 0)
	colors = ['#8fce00', '#ff1100']
	dem = df[df['final_dx'] == 'Alzheimer Disease']
	nondem = df[df['final_dx'] == 'Cognitively Normal']
	features = {
		'sex': ['Male', 'Female'],
		'age': ['81-96', '65-80', '46-65'],
		'bmi': ['31-60', '26-30', '11-25'],
		'boston_naming_test': ['41-60', '21-40', '00-20']
	}
	fig, axs = plt.subplots(4, 4, figsize=(18, 24))
	fig.subplots_adjust(wspace=0)
	for i, axl in enumerate(axs):
		key = list(features.keys())[i]
		data = [
			[len(dem[dem[key] == f]) if key == 'sex' else len(dem[(dem[key] >= int(f.split('-')[0])) & (dem[key] <= int(f.split('-')[1]))]) for f in features[key]],
			[len(nondem[nondem[key] == f]) if key == 'sex' else len(nondem[(nondem[key] >= int(f.split('-')[0])) & (nondem[key] <= int(f.split('-')[1]))]) for f in features[key]]
		]
		_plot_bar_of_pie(
			axs=[axl[0], axl[1]],
			labels=[['AD', 'Non-AD'], features[key]],
			data=[[len(dem), len(nondem)], data[0]],
			title='AD by ' + key,
			angle=-220,
			colors=[colors[1], colors[0]]
		)
		_plot_bar_of_pie(
			axs=[axl[2], axl[3]],
			labels=[['Non-AD', 'AD'], features[key]],
			data=[[len(nondem), len(dem)], data[1]],
			title='Non-AD by ' + key,
			angle=40,
			colors=[colors[0], colors[1]]
		)
	fig.suptitle('\n\nANALYSIS OF NUMERICAL FEATURES', fontsize=20, fontweight='bold')
	plt.show()


def categorical_features(folder):
	"""
	Plots categorical features.
	Args:
		folder (str): the path of the folder containing csv files.
	Returns:
		None.
	"""
	fig, axs = plt.subplots(5, 3, figsize=(18, 22))
	df = pd.read_csv(os.path.join(folder, 'data_desc.csv'))
	counter = 0
	features_to_plot = ['ethnicity', 'education', 'marriage', 'smoke', 'brain_disease', 'heart_disease', 'cdr_memory', 'cdr_orientation', 'cdr_judgment', 'cdr_community', 'cdr_hobbies', 'cdr_personalcare', 'depression', 'sleeping_disorder', 'motor_disturbance']
	for axl in axs:
		for ax in axl:
			labels = df[features_to_plot[counter]].unique()
			data = {
				'AD': [len(df[(df['final_dx'] == 'Alzheimer Disease') & (df[features_to_plot[counter]] == l)]) for l in labels],
				'Non-AD': [len(df[(df['final_dx'] == 'Cognitively Normal') & (df[features_to_plot[counter]] == l)]) for l in labels]
			}
			x1 = np.arange(len(labels))
			width = 0.25
			multiplier = 0
			for attribute, measurement in data.items():
				offset = width * multiplier
				rects = ax.bar(x1 + offset, [round(i, 3) for i in measurement], width, label = attribute, color = ['#ff1100' if attribute == 'AD' else '#8fce00'])
				ax.bar_label(rects, padding = 0, fontsize = 14)
				multiplier += 1
			ax.set_title(features_to_plot[counter], fontsize=18)
			ax.set_xticks(x1 + (width / 2), labels)
			ax.legend(fontsize=14)
			counter += 1
	fig.suptitle('ANALYSIS OF CATEGORICAL FEATURES\n\n', fontsize=20, fontweight='bold')
	fig.tight_layout()
	plt.show()


def confusion_matrix(folder):
	"""
	Plots confusion matrix.
	Args:
		folder (str): the path of the folder containing csv files.
	Returns:
		None.
	"""
	df = pd.read_csv(os.path.join(folder, 'data_num.csv'))
	columns = ['sex', 'age', 'ethnicity', 'education', 'marriage', 'weight', 'height', 'smoke', 'brain_disease', 'heart_disease', 'cdr_memory', 'cdr_orientation', 'cdr_judgment', 'cdr_community', 'cdr_hobbies', 'cdr_personalcare', 'boston_naming_test', 'depression', 'sleeping_disorder', 'motor_disturbance']
	df = df[columns]
	fig, ax = plt.subplots(figsize=(18, 10))
	ax.set_title('FEATURES CORRELATION\n', fontsize=20, fontweight='bold')
	sns.heatmap(df.corr(), linewidths=.5, fmt='g', cmap='viridis', ax=ax)
	ax.tick_params(axis='both', which='major', labelsize=14)
	fig.tight_layout()
	plt.show()


def _plot_bar_of_pie(axs, labels, data, title, angle, colors):
	"""
	Plots a single bar of pie
	(See https://matplotlib.org/stable/gallery/pie_and_polar_charts/bar_of_pie.html).
	Args:
		axs (list): list containing the left axes (pie chart) and the right axes (bar chart).
		labels (list): list containing the pie chart and bar chart labels as list itself.
		data (list): list containing the pie chart and bar chart data as list itself.
		title (str): the title of the bar of pie.
		angle (int): angle of rotation of the pie.
		colors (list): list of colors.
	Returns:
		None.
	"""
	wedges, *_ = axs[0].pie(data[0], autopct='%1.1f%%', startangle=(angle * data[0][0]), labels=[l + ' ('+str(data[0][i])+')' for i, l in enumerate(labels[0])], explode=[0.1, 0], colors=colors, textprops={'fontsize': 14})
	bottom = 1
	width = .2
	for j, (height, label) in enumerate(reversed([*zip(data[1], labels[1])])):
		bottom -= height
		bc = axs[1].bar(0, height, width, bottom=bottom, color=colors[0], label=label, alpha=0.1 + 0.25 * j)
		axs[1].bar_label(bc, labels=[f"{height}"], label_type='center', fontsize=14)

	axs[0].set_title(title, fontsize=18)
	axs[1].legend(fontsize=14)
	axs[1].axis('off')
	axs[1].set_xlim(- 2.5 * width, 2.5 * width)

	# use ConnectionPatch to draw lines between the two plots
	theta1, theta2 = wedges[0].theta1, wedges[0].theta2
	center, r = wedges[0].center, wedges[0].r
	bar_height = sum(data[1])

	# draw top connecting line
	x = r * np.cos(np.pi / 180 * theta2) + center[0]
	y = r * np.sin(np.pi / 180 * theta2) + center[1]
	con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=axs[1].transData, xyB=(x, y), coordsB=axs[0].transData)
	con.set_color([0, 0, 0])
	con.set_linewidth(1)
	axs[1].add_artist(con)

	# draw bottom connecting line
	x = r * np.cos(np.pi / 180 * theta1) + center[0]
	y = r * np.sin(np.pi / 180 * theta1) + center[1]
	con = ConnectionPatch(xyA=(-width / 2, -bar_height), coordsA=axs[1].transData, xyB=(x, y), coordsB=axs[0].transData)
	con.set_color([0, 0, 0])
	axs[1].add_artist(con)
	con.set_linewidth(1)


def training_values(folder):
	"""
	Plots losses and metrics over training phase.
	Args:
		folder (str): the path of the folder containing the csv reports.
	Returns:
		None.
	"""
	try:
		n = 'DenseNetMM_training.csv'
		df = pd.read_csv(os.path.join(folder, n))
		run_id = sorted(df['id'].unique())[-1]
		data_df = df[df['id'] == run_id]
		best_epoch = data_df.iloc[data_df['auc_score'].idxmax()]['epoch']
		x = [i + 1 for i in range(len(data_df))]
		fig, ax = plt.subplots(1, 1, figsize=(18, 6))
		ax.plot(x, data_df['train_crossentropy_loss'].to_numpy(), label='training_loss')
		ax.plot(x, data_df['eval_crossentropy_loss'].to_numpy(), label='evaluation_loss')
		ax.set_xticks([i for i in range(0, len(data_df), 5)])
		plt.axvline(best_epoch, linewidth=4, color='red', label='best_run')
		plt.xlabel('EPOCHS', fontsize=14)
		plt.ylabel('CROSSENTROPY LOSS', fontsize=14)
		plt.title('TRAINING LOSSES', fontsize=18)
		plt.legend(fontsize=14)
		fig.tight_layout()
		plt.show()
		fig, ax = plt.subplots(1, 1, figsize=(18, 6))
		ax.plot(x, data_df['auc_score'].to_numpy(), label='auc_score')
		ax.set_xticks([i for i in range(0, len(data_df), 5)])
		plt.axvline(best_epoch, linewidth=4, color='red', label='best_run')
		plt.xlabel('EPOCHS', fontsize=14)
		plt.ylabel('AUC SCORE', fontsize=14)
		plt.title('TRAINING METRICS', fontsize=18)
		plt.legend(fontsize=14)
		fig.tight_layout()
		plt.show()
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: model report for\033[95m '+n+'\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def image_transformations(data, channel='T1w', size=128, session=False):
	"""
	Plots a sample image before and after preprocessing transformations.
	Args:
		data (list): data set according to `monai.data.Dataset` format.
		channel (str): the MR channel. Possible values are `T1w` or `T2w`.
		size (): size for the input image. Final input shape will be (`size`, `size`, `size`)
		session (bool | str): a specific session to plot. If `False` a random sample will be selected.
	Returns:
		None.
	"""
	warnings.filterwarnings('ignore')
	if session:
		s = [i for i in data if i['image'][0].split('/')[-2] == session]
		if len(s):
			s = s[-1]
			subject_id = s['image'][0].split('/')[-2]
		else:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: session if\033[95m '+session+'\033[0m not found on sample.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
	else:
		s = random.sample(data, 1)[0]
		subject_id = s['image'][0].split('/')[-2]
	train_transform, _ = get_transformations(size)
	trans = train_transform(s)
	before = nib.load(s['image'][0]) if channel == 'T1w' else nib.load(s['image'][1])
	after = nib.Nifti1Image(trans['image'][0].numpy(), affine=np.eye(4)) if channel == 'T1w' else nib.Nifti1Image(trans['image'][1].numpy(), affine=np.eye(4))
	_, axs = plt.subplots(2, 1, figsize=(18, 10))
	plotting.plot_anat(before, display_mode='ortho', axes=axs[0], title='Session: ' + subject_id + ' Shape: ' + str(before.shape) + ' Channel: ' + channel + ' - BEFORE')
	plotting.plot_anat(after, display_mode='ortho', axes=axs[1], title='Session: ' + subject_id + ' Shape: ' + str(after.shape) + ' Channel: ' + channel + ' - AFTER')
	plt.show()


def results(folder):
	"""
	Plots all metrics calculated over the testing set.
	Args:
		folder (str): the path of the folder containing the csv reports.
	Returns:
		None.
	"""
	channels = ['T1w+T2w', 'T1w', 'T2w', 'T2w', 'T2w+feat', 'T1w+T2w+feat', 'T2w+demo']
	sizes = [128, 128, 128, 256, 128, 128, 128]
	try:
		p = os.path.join(folder, 'results.csv')
		df = pd.read_csv(p)
		print(''.join(['> ' for i in range(45)]))
		print(f'\n{"CHANNEL":<14}{"SIZE":<8}{"ACCURACY":<12}{"PRECISION":<12}{"RECALL":<12}{"F1":<12}{"ROC AUC":<12}{"AVG":<12}\n')
		print(''.join(['> ' for i in range(45)]))
		counter = 0
		for i in range(len(df)):
			if math.isnan(df.iloc[i]['accuracy_score_2']):
				a = df.iloc[i]['accuracy_score']
				p = df.iloc[i]['precision_score']
				r = df.iloc[i]['recall_score']
				f = df.iloc[i]['f1_score']
				ra = df.iloc[i]['roc_auc_score']
				if i == 4:
					print(f'\033[1m{channels[counter]:<14}{sizes[counter]:<8}{a:<12.4f}{p:<12.4f}{r:<12.4f}{f:<12.4f}{ra:<12.4f}{(a+p+r+f+ra)/5:<12.4f}\033[0m')
				else:
					print(f'{channels[counter]:<14}{sizes[counter]:<8}{a:<12.4f}{p:<12.4f}{r:<12.4f}{f:<12.4f}{ra:<12.4f}{(a+p+r+f+ra)/5:<12.4f}')
				counter += 1
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: file\033[95m  reports.csv\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def results_multiclass(folder):
	"""
	Plots all metrics calculated over the testing set for each class.
	Args:
		folder (str): the path of the folder containing the csv reports.
	Returns:
		None.
	"""
	try:
		metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
		channels = ['T2w', 'T2w+demo']
		p = os.path.join(folder, 'results.csv')
		df = pd.read_csv(p)
		print(''.join(['> ' for i in range(48)]))
		print(f'\n{"CHANNEL":<12}{"METRIC":<9}{"COGNITIVE_NORMAL":>20}{"EARLY_STAGE_AD":>18}{"ALZHEIMER_DISEASE":>20}\033[1m{"AVERAGE":>12}\033[0m\n')
		print(''.join(['> ' for i in range(48)]))
		counter = 0
		for i in range(len(df)):
			if not math.isnan(df.iloc[i]['accuracy_score_2']):
				avgs = {'0': 0, '1': 0, '2': 0, 'avg': 0}
				for k, m in enumerate(metrics):
					m_avg = df.iloc[i][m + '_score']
					m_0 = df.iloc[i][m + '_score_0']
					m_1 = df.iloc[i][m + '_score_1']
					m_2 = df.iloc[i][m + '_score_2']
					avgs['avg'] += m_avg
					avgs['0'] += m_0
					avgs['1'] += m_1
					avgs['2'] += m_2
					print(f'{channels[counter]:<12}{m.upper():<9}{m_0:>20.4f}{m_1:>18.4f}{m_2:>20.4f}\033[1m{m_avg:>12.4f}\033[0m')
				print(''.join(['-' for i in range(96)]))
				print(f'\033[1m{channels[counter]:<12}{"AVG":<9}{(avgs["0"] / len(metrics)):>20.4f}{(avgs["1"] / len(metrics)):>18.4f}{(avgs["2"] / len(metrics)):>20.4f}{(avgs["avg"] / len(metrics)):>12.4f}\033[0m')
				print('\n')
				counter += 1
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: file\033[95m  reports.csv\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def gradcam(image, label, pred, heatmap, mask, alpha=128):
	"""
	Plots model input image, Grad-CAM heatmap and segmentation mask.
	Args:
		image (numpy.ndarray): the input 3D image.
		label (int): the input image label.
		pred (int): model prediction for input image.
		heatmap (numpy.ndarray): the Grad-CAM 3D heatmap.
		mask (numpy.ndarray): the computed 3D segmentation mask.
		alpha (int): transparency channel. Between 0 and 255.
	Returns:
		None.
	"""
	if alpha >= 0 and alpha <= 255:
		heatmap_mask = np.zeros((image.shape[0], image.shape[1], image.shape[2], 4), dtype='uint8')
		heatmap_mask[mask == 1] = [255, 0, 0, alpha]
		image = image[:,:,int(image.shape[2] / 2)]
		heatmap = heatmap[:,:,int(heatmap.shape[2] / 2)]
		heatmap_mask = heatmap_mask[:,:,int(heatmap_mask.shape[2] / 2),:]
		fig, axs = plt.subplots(1, 3, figsize=(18, 6))
		norm_img = cv2.normalize(image, np.zeros((image.shape[1], image.shape[0])), 0, 1, cv2.NORM_MINMAX)
		im_shows = [
			axs[0].imshow(norm_img, cmap='gray', interpolation='bilinear', vmin = .0, vmax = 1.),
			axs[1].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin = .0, vmax = 1.),
			axs[2].imshow(norm_img, cmap='gray', interpolation='bilinear', vmin = .0, vmax = 1.)
		]
		axs[2].imshow(heatmap_mask, interpolation='bilinear')
		axs[0].set_title('Label=' + ('NON-AD' if label == 0 else 'AD') + ' | Prediction=' + ('NON-AD' if pred == 0 else 'AD'), fontsize=16)
		axs[1].set_title('Grad-CAM Heatmap', fontsize=16)
		axs[2].set_title('Mask - Threshold ' + str(.8), fontsize=16)
		for i, ax in enumerate(axs):
			ax.axis('off')
			fig.colorbar(im_shows[i], ax=ax, ticks=np.linspace(0,1,6))
		fig.tight_layout()
		plt.show()
	else:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: alpha channel \033[95m '+alpha+'\033[0m out of range [0,255].\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def available_llms():
	"""
	Printing all available llms as defined in `src.helpers.config`.
	Args:
		None.
	Returns:
		None.
	"""
	_config = get_config()
	llms = _config.get('LLM')
	print(''.join(['> ' for i in range(35)]))
	print(f'\n{"MODEL_KEY":<15}{"MODEL_NAME":<25}\n')
	print(''.join(['> ' for i in range(35)]))
	for k in llms.keys():
		print(f'\033[1m{k:<15}\033[0m{llms[k].split("/")[-1]:<25}')


def llms_metrics(report_path):
	"""
	Plot metrics related to LLMs.
	Args:
		report_path (str): absolute path where metrics data are saved.
	Returns:
		None.
	"""
	_config = get_config()
	llm_params = _config.get('LLM_PARAMS')
	llm_params = {k: int(v[:-1]) for k, v in llm_params.items()}
	llm_params = {k: v for k, v in sorted(llm_params.items(), key = lambda item: item[1], reverse = False)}
	df = pd.read_csv(os.path.join(report_path, 'LLM_metrics.csv'))
	llm_times = df.loc[:, df.columns != 'lang'].groupby('model').mean()['inference_time'].to_dict()
	llm_times = {k: v for k, v in sorted(llm_times.items(), key = lambda item: item[1], reverse = False)}
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
	rects = ax1.barh(list(llm_params.keys()), list(llm_params.values()), align='center', height=0.5)
	ax1.bar_label(rects, [str(i) + 'B' for i in list(llm_params.values())], padding=-30, color='white', fontsize=16, fontweight='bold')
	ax1.set_xticks(np.arange(0, max(list(llm_params.values())) + 1))
	ax1.set_yticks(ticks=list(llm_params.keys()), labels=list(llm_params.keys()), fontsize=14)
	ax1.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
	ax1.set_title('Number of params', fontsize=20, fontweight='bold')
	ax1.set_xlabel('Number of params (in billions)', fontsize=14)
	rects = ax2.barh(list(llm_times.keys()), list(llm_times.values()), align='center', height=0.5)
	ax2.bar_label(rects, [str(int(i)) + 's' for i in list(llm_times.values())], padding=-48, color='white', fontsize=16, fontweight='bold')
	ax2.set_xticks(np.arange(0, max(list(llm_times.values())) + 10, 50))
	ax2.set_yticks(ticks=list(llm_times.keys()), labels=list(llm_times.keys()), fontsize=14)
	ax2.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
	ax2.set_title('Inference times', fontsize=20, fontweight='bold')
	ax2.set_xlabel('Inference times (in seconds)', fontsize=14)
	fig.tight_layout()
	plt.show()


def llms_textual_metrics(metrics, titles, report_path, dynamic_padding_rate = 10):
	"""
	Plot metrics related to LLMs textual outputs.
	Args:
		metrics (list): two numerical metrics to plot included in the report file.
		titles (list): two titles to associate to the selected metrics.
		report_path (str): absolute path where metrics data are saved.
		dynamic_padding_rate (int): plotted lines padding expressed as percentage.
	Returns:
		None.
	"""
	df = pd.read_csv(os.path.join(report_path, 'LLM_metrics.csv'))
	llm_en = ['biomistral', 'llama', 'mistral']
	llm_it = ['llama', 'mistral', 'llamantino2', 'llamantino3', 'minerva']
	left_ylabels = ['English', 'Italian']
	left_score = df.groupby(['model', 'lang', 'prompt_id']).mean()[metrics[0]].to_dict()
	left_score_en = [[v for k, v in left_score.items() if k[0] == m and k[1] == 'EN' ] for m in llm_en]
	left_score_it = [[v for k, v in left_score.items() if k[0] == m and k[1] == 'IT' ] for m in llm_it]
	right_score = df.groupby(['model', 'lang', 'prompt_id']).mean()[metrics[1]].to_dict()
	right_score_en = [[v for k, v in right_score.items() if k[0] == m and k[1] == 'EN' ] for m in llm_en]
	right_score_it = [[v for k, v in right_score.items() if k[0] == m and k[1] == 'IT' ] for m in llm_it]

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
	axs = [ax1, ax2, ax3, ax4]
	data = [left_score_en, right_score_en, left_score_it, right_score_it]
	for j, ax in enumerate(axs):
		llms = llm_en if j < 2 else llm_it
		ticks = [i for i in range(len(llms))]
		ax.plot(ticks, data[j], label=['Prompt_1', 'Prompt_2'], marker='D', markersize=12, linewidth=3)
		ax.plot(ticks, [np.mean(i) for i in data[j]], label='mean', color='red', marker='o', markersize=8, linestyle='dotted', linewidth=2)
		ax.set_xticks(ticks=ticks, labels=llms, fontsize=14)
		ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
		ax.legend(fontsize=12, labelspacing=1)
		if j < 2:
			ax.set_title(titles[j], fontsize=20, fontweight='bold', pad=20)
		if j % 2 == 0:
			ax.set_ylabel(left_ylabels[int(j/2)], fontsize=16, fontweight='bold')
			dyn_pad = (max(left_score.values()) - min(left_score.values())) * dynamic_padding_rate / 100
			ax.set_ylim([min(left_score.values()) - dyn_pad, max(left_score.values()) + dyn_pad])
		else:
			dyn_pad = (max(right_score.values()) - min(right_score.values())) * dynamic_padding_rate / 100
			ax.set_ylim([min(right_score.values()) - dyn_pad, max(right_score.values()) + dyn_pad])
	fig.tight_layout()
	plt.show()


def llms_average_metrics(report_path, lang = 'all'):
	"""
	Plot all metrics by averaging their values.
	Args:
		report_path (str): absolute path where metrics data are saved.
		lang (str): the language by which aggregate the results. Deafult `all` do not filter by language.
	Returns:
		None.
	"""
	df = pd.read_csv(os.path.join(report_path, 'LLM_metrics.csv'))
	metrics = [m for m in df.columns if m not in ['model', 'lang', 'prompt_id']]
	llms = {
		'en': ['biomistral', 'llama', 'mistral'],
		'it': ['llama', 'mistral', 'llamantino2', 'llamantino3', 'minerva'],
		'all': ['biomistral', 'llama', 'mistral', 'llamantino2', 'llamantino3', 'minerva']
	}
	print(''.join(['> ' for i in range(55)]))
	if lang.lower() == 'en':
		print(f'\n{"METRIC":<24}{"BIOMISTRAL":>14}{"LLAMA":>14}{"MISTRAL":>14}\n')
	elif lang.lower() == 'it':
		print(f'\n{"METRIC":<24}{"LLAMA":>14}{"MISTRAL":>14}{"LLAMANTINO2":>14}{"LLAMANTINO3":>14}{"MINERVA":>14}\n')
	else:
		print(f'\n{"METRIC":<24}{"BIOMISTRAL":>14}{"LLAMA":>14}{"MISTRAL":>14}{"LLAMANTINO2":>14}{"LLAMANTINO3":>14}{"MINERVA":>14}\n')
	print(''.join(['> ' for i in range(55)]))
	for m in metrics:
		if lang.lower() == 'all':
			data = [df.loc[df['model'] == l][m].mean() for l in llms[lang.lower()]]
		else:
			data = [df.loc[(df['model'] == l) & (df['lang'] == lang.upper())][m].mean() for l in llms[lang.lower()]]
		s = ''
		for k, d in enumerate(data):
			ds = str(round(d, 2))
			s += ''.join([' ' for i in range(14 - len(ds))])
			if m == 'inference_time' or m == 'diversity_MAAS':
				s += ('\033[1m\033[91m'+ds+'\033[0m' if k == np.argmax(data) else ('\033[1m\033[92m'+ds+'\033[0m' if k ==  np.argmin(data) else ds))
			else:
				s += ('\033[1m\033[92m'+ds+'\033[0m' if k == np.argmax(data) else ('\033[1m\033[91m'+ds+'\033[0m' if k ==  np.argmin(data) else ds))
		print(f'{m:<24}{s:>14}')
