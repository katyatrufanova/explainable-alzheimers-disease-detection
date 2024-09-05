"""
Functions for training the models.
"""
import os, random, time, calendar, datetime, warnings
from sys import platform
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.helpers.utils import get_date_time, save_results


__all__ = ['train_test_splitting', 'training_model', 'predict_model']


def train_test_splitting(
		data_folder,
		meta_folder,
		channels,
		features,
		train_ratio=.8,
		multiclass=False,
		reports_path=None,
		write_to_file=False,
		load_from_file=False,
		verbose=True
	):
	"""
	Splitting train/eval/test.
	Args:
		data_folder (str): path of the folder containing images.
		meta_folder (str): path of the folder containing csv files.
		channels (list): image channels to select (values `T1w`, `T2w` or both).
		features (list): features set to select.
		train_ratio (float): ratio of the training set, value between 0 and 1.
		multiclass (bool): `False` for binary classification, `True` for ternary classification.
		reports_path (str): folder where to save report file.
			Required if `write_to_file` is True and/or `load_from_file` is True.
		write_to_file (bool): whether to write selected data to csv file.
		load_from_file (bool): whether to load splitting data from a previous saved csv file.
		verbose (bool): whether or not print information.
	Returns:
		train_data (list): the training data ready to feed monai.data.Dataset
		eval_data (list): the evaluation data ready to feed monai.data.Dataset
		test_data (list): the testing data ready to feed monai.data.Dataset.
		(see https://docs.monai.io/en/latest/data.html#monai.data.Dataset).
	"""
	# preparing numerical data and utils
	scaler = MinMaxScaler()
	df = pd.read_csv(os.path.join(meta_folder, 'data_num.csv'))
	df1 = df[(df['weight'] != .0) & (df['height'] != .0)]
	df['bmi'] = round(df1['weight'] / (df1['height'] * df1['height']), 0)
	df['bmi'] = df['bmi'].fillna(.0)
	sessions = [s.split('_')[0] for s in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, s))]
	subjects = list(set(sessions))

	# applying splitting on subjects to prevent data leakage
	random.shuffle(subjects)
	split_train = int(len(subjects) * train_ratio)
	train_subjects, test_subjects = subjects[:split_train], subjects[split_train:]
	split_eval = int(len(train_subjects) * .8)
	eval_subjects = train_subjects[split_eval:]
	train_subjects = train_subjects[:split_eval]

	# applying multiclass label correction and splitting
	if multiclass:
		train_subjects, eval_subjects, test_subjects = [], [], []
		df.loc[df['cdr'] == .0, 'final_dx'] = .0
		df.loc[df['cdr'] == .5, 'final_dx'] = 1.
		df.loc[(df['cdr'] != .0) & (df['cdr'] != .5), 'final_dx'] = 2.
		m = np.min(np.unique(df['final_dx'].to_numpy(), return_counts=True)[1])
		df = pd.concat([
			df[df['final_dx'] == .0].sample(m),
			df[df['final_dx'] == 1.].sample(m),
			df[df['final_dx'] == 2.].sample(m)
		], ignore_index=True)
		n_test = m - int(m * train_ratio)
		n_eval = m - n_test - int(m * train_ratio * train_ratio)
		for i in range(3):
			sub = list(set(df[df['final_dx'] == float(i)]['subject_id'].to_numpy()))
			random.shuffle(sub)
			counter = 0
			for j in range(len(sub)):
				counter += len(df[df['subject_id'] == sub[j]])
				if counter <= n_test:
					test_subjects.append(sub[j])
				elif counter > n_test and counter <= (n_test + n_eval):
					eval_subjects.append(sub[j])
				else:
					train_subjects.append(sub[j])

	# loading sessions paths
	X_train = df[df['subject_id'].isin(train_subjects)]
	X_eval = df[df['subject_id'].isin(eval_subjects)]
	X_test = df[df['subject_id'].isin(test_subjects)]
	train_sessions = [os.path.join(data_folder, s) for s in X_train['session_id'].values]
	eval_sessions = [os.path.join(data_folder, s) for s in X_eval['session_id'].values]
	test_sessions = [os.path.join(data_folder, s) for s in X_test['session_id'].values]

	# writing splitting to file
	if write_to_file:
		if not reports_path:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: Paremeter \033[95m `reports_path`\033[0m must be specified.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
			return [],[],[]
		else:
			for i in range(max(len(train_sessions), len(eval_sessions), len(test_sessions))):
				save_results(
					os.path.join(reports_path, 'splitting'+('_mc_' if multiclass else '_bin_')+str(calendar.timegm(time.gmtime()))+'.csv'),
					{
						'train_sessions': train_sessions[i].split('/')[-1] if i < len(train_sessions) else '',
						'eval_sessions': eval_sessions[i].split('/')[-1] if i < len(eval_sessions) else '',
						'test_sessions': test_sessions[i].split('/')[-1] if i < len(test_sessions) else ''
					}
				)

	# loading (the latest) splitting from file
	if load_from_file:
		if not reports_path:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: Paremeter \033[95m `reports_path`\033[0m must be specified.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
			return [],[],[]
		else:
			k = 'splitting_mc_' if multiclass else 'splitting_bin_'
			df_split = pd.read_csv(os.path.join(reports_path, sorted([i for i in os.listdir(reports_path) if k in i])[-1]))
			df = pd.read_csv(os.path.join(meta_folder, 'data_num.csv'))
			df = df[df['session_id'].isin((df_split['train_sessions'].dropna().to_list() + df_split['eval_sessions'].dropna().to_list() + df_split['test_sessions'].dropna().to_list()))]
			df1 = df[(df['weight'] != .0) & (df['height'] != .0)]
			df['bmi'] = round(df1['weight'] / (df1['height'] * df1['height']), 0)
			df['bmi'] = df['bmi'].fillna(.0)
			if multiclass:
				df.loc[df['cdr'] == .0, 'final_dx'] = .0
				df.loc[df['cdr'] == .5, 'final_dx'] = 1.
				df.loc[(df['cdr'] != .0) & (df['cdr'] != .5), 'final_dx'] = 2.
			X_train = df[df['session_id'].isin(df_split['train_sessions'].dropna().to_numpy())]
			X_eval = df[df['session_id'].isin(df_split['eval_sessions'].dropna().to_numpy())]
			X_test = df[df['session_id'].isin(df_split['test_sessions'].dropna().to_numpy())]
			train_sessions = [os.path.join(data_folder, s) for s in X_train['session_id'].values]
			eval_sessions = [os.path.join(data_folder, s) for s in X_eval['session_id'].values]
			test_sessions = [os.path.join(data_folder, s) for s in X_test['session_id'].values]
			train_subjects = list(set([s.split('_')[0] for s in df_split['train_sessions'].dropna().to_numpy()]))
			eval_subjects = list(set([s.split('_')[0] for s in df_split['eval_sessions'].dropna().to_numpy()]))
			test_subjects = list(set([s.split('_')[0] for s in df_split['test_sessions'].dropna().to_numpy()]))

	# scaling numerical data in range [0,1]
	X_train.loc[:, features] = scaler.fit_transform(X_train[features])
	X_eval.loc[:, features] = scaler.fit_transform(X_eval[features])
	X_test.loc[:, features] = scaler.fit_transform(X_test[features])

	# arranging data in dictionaries
	train_data, eval_data, test_data = {}, {}, {}
	train_data = [dict({
		'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
		'data': X_train[X_train['session_id'] == s.split('/')[-1]][features].values[0],
		'label': df[df['session_id'] == s.split('/')[-1]]['final_dx'].values[0]
	}) for s in train_sessions]
	eval_data = [dict({
		'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
		'data': X_eval[X_eval['session_id'] == s.split('/')[-1]][features].values[0],
		'label': df[df['session_id'] == s.split('/')[-1]]['final_dx'].values[0]
	}) for s in eval_sessions]
	test_data = [dict({
		'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
		'data': X_test[X_test['session_id'] == s.split('/')[-1]][features].values[0],
		'label': df[df['session_id'] == s.split('/')[-1]]['final_dx'].values[0]
	}) for s in test_sessions]

	# print data splitting information
	if verbose:
		print(''.join(['> ' for i in range(40)]))
		print(f'\n{"":<20}{"TRAINING":<20}{"EVALUATION":<20}{"TESTING":<20}\n')
		print(''.join(['> ' for i in range(40)]))
		tsb1 = str(len(train_subjects)) + ' (' + str(round((len(train_subjects) * 100 / len(df['subject_id'].unique())), 0)) + ' %)'
		tsb2 = str(len(eval_subjects)) + ' (' + str(round((len(eval_subjects) * 100 / len(df['subject_id'].unique())), 0)) + ' %)'
		tsb3 = str(len(test_subjects)) + ' (' + str(round((len(test_subjects) * 100 / len(df['subject_id'].unique())), 0)) + ' %)'
		tss1 = str(len(train_sessions)) + ' (' + str(round((len(train_sessions) * 100 / len(df)), 2)) + ' %)'
		tss2 = str(len(eval_sessions)) + ' (' + str(round((len(eval_sessions) * 100 / len(df)), 2)) + ' %)'
		tss3 = str(len(test_sessions)) + ' (' + str(round((len(test_sessions) * 100 / len(df)), 2)) + ' %)'
		print(f'\n{"subjects":<20}{tsb1:<20}{tsb2:<20}{tsb3:<20}\n')
		print(f'{"sessions":<20}{tss1:<20}{tss2:<20}{tss3:<20}\n')
	return train_data, eval_data, test_data


def training_model(
		model,
		data,
		transforms,
		epochs,
		device,
		paths,
		batch_size=10,
		val_interval=1,
		early_stopping=10,
		num_workers=4,
		ministep=14,
		write_to_file=True,
		verbose=False
	):
	"""
	Standard Pytorch-style training program.
	Args:
		model (torch.nn.Module): the model to be trained.
		data (list): the training and evalutaion data.
		transform (list): transformation sequence for training and evaluation data.
		epochs (int): max number of epochs.
		device (str): device's name.
		paths (list): folders where to save results and model's dump.
		batch_size (int): size of the batches.
		val_interval (int): validation interval.
		early_stopping (int): nr. of epochs for those there's no more improvements.
		num_workers (int): setting multi-process data loading.
		ministep (int): number of interval of data to load on RAM.
		write_to_file (bool): whether to write results to csv file.
		verbose (bool): whether to print minimal or extended information.
	Returns:
		metrics (list): the list of all the computed metrics over the training in this order:
			- dice loss during training;
			- dice loss during evaluation;
			- execution times;
			- average dice score;
			- dice score for the class ET;
			- dice score for the class TC;
			- dice score for the class WT.
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	train_data, eval_data = data
	train_transform, eval_transform = transforms
	saved_path, reports_path, logs_path = paths
	ministep = ministep if (len(train_data) > 10 and len(eval_data) > 10 and ministep > 1) else 2
	post_pred = Compose([Activations(softmax=True)])
	post_label = Compose([AsDiscrete(to_onehot=model.out_channels)])
	warnings.filterwarnings('ignore')

	# define CrossEntropy loss, Adam optimizer, AUC metric, Cosine Annealing scheduler
	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
	acc_metric = ROCAUCMetric()
	scaler = torch.cuda.amp.GradScaler() # use Automatic Pixed Precision to accelerate training
	torch.backends.cudnn.benchmark = True # enable cuDNN benchmark

	# define metric/loss collectors
	best_metric, best_metric_epoch = -1, -1
	best_metrics_epochs_and_time = [[], [], []]
	epoch_loss_values, epoch_time_values, metric_values = [[], []], [], []

	# log the current execution
	log = open(os.path.join(logs_path, 'training.log'), 'a', encoding='utf-8')
	log.write('['+get_date_time()+'] Training phase started.EXECUTING: ' + model.name + '\n')
	log.flush()
	ts = calendar.timegm(time.gmtime())
	total_start = time.time()
	for epoch in range(epochs):
		epoch_start = time.time()
		print(''.join(['> ' for i in range(40)]))
		print(f"epoch {epoch + 1}/{epochs}")
		log.write('['+get_date_time() + '] EXECUTING.' + model.name + ' EPOCH ' + str(epoch + 1) + ' OF ' + str(epochs) + ' \n')
		log.flush()
		model.train()
		epoch_loss_train, epoch_loss_eval = 0, 0
		step_train, step_eval = 0, 0
		ministeps_train = np.linspace(0, len(train_data), ministep).astype(int)
		ministeps_eval = np.linspace(0, len(eval_data), ministep).astype(int)

		# start training
		for i in range(len(ministeps_train) - 1):
			train_ds = CacheDataset(train_data[ministeps_train[i]:ministeps_train[i+1]], transform=train_transform, cache_rate=1.0, num_workers=None, progress=False)
			train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
			for batch_data in train_loader:
				step_start = time.time()
				step_train += 1
				inputs_img, inputs_data, labels = (
					batch_data['image'].to(device),
					batch_data['data'].to(device),
					batch_data['label'].to(device)
				)
				optimizer.zero_grad()
				with torch.cuda.amp.autocast():
					outputs = model([inputs_img, inputs_data])
					loss = loss_function(outputs, labels)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				epoch_loss_train += loss.item()
				if verbose:
					print(
						f"{step_train}/{len(train_data) // train_loader.batch_size}"
						f", train_loss: {loss.item():.4f}"
						f", step time: {str(datetime.timedelta(seconds=int(time.time() - step_start)))}"
					)
		lr_scheduler.step()
		epoch_loss_train /= step_train
		epoch_loss_values[0].append(epoch_loss_train)
		print(f"epoch {epoch + 1} average training loss: {epoch_loss_train:.4f}")

		# start validation
		if (epoch + 1) % val_interval == 0:
			model.eval()
			with torch.no_grad():
				y_pred = torch.tensor([], dtype=torch.float32, device=device)
				y = torch.tensor([], dtype=torch.long, device=device)
				for i in range(len(ministeps_eval) - 1):
					eval_ds = CacheDataset(eval_data[ministeps_eval[i]:ministeps_eval[i+1]], transform=eval_transform, cache_rate=1.0, num_workers=None, progress=False)
					eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
					for val_data in eval_loader:
						step_eval += 1
						val_inputs_img, val_inputs_data, val_label = (
							val_data['image'].to(device),
							val_data['data'].to(device),
							val_data['label'].to(device)
						)
						val_outputs = model([val_inputs_img, val_inputs_data])
						y_pred = torch.cat([y_pred, val_outputs], dim=0)
						y = torch.cat([y, val_label], dim=0)
						val_loss = loss_function(val_outputs, val_label)
						epoch_loss_eval += val_loss.item()
				y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
				y_pred_act = [post_pred(i) for i in decollate_batch(y_pred, detach=False)]
				acc_metric(y_pred=y_pred_act, y=y_onehot)
				epoch_loss_eval /= step_eval
				epoch_loss_values[1].append(epoch_loss_eval)

				# calculate metrics
				metric = acc_metric.aggregate().item()
				metric_values.append(metric)
				acc_metric.reset()

				# save best performing model
				if metric > best_metric:
					best_metric = metric
					best_metric_epoch = epoch + 1
					best_metrics_epochs_and_time[0].append(best_metric)
					best_metrics_epochs_and_time[1].append(best_metric_epoch)
					best_metrics_epochs_and_time[2].append(time.time() - total_start)
					torch.save(model.state_dict(), os.path.join(saved_path, model.name + '_best.pth'))
					print("saved new best model")
				print(
					f"current epoch: {epoch + 1} current mean AUC: {metric:.4f}"
					f"\nbest mean AUC: {best_metric:.4f}"
					f" at epoch: {best_metric_epoch}"
				)
		print(f"time consuming of epoch {epoch + 1} is: {str(datetime.timedelta(seconds=int(time.time() - epoch_start)))}")
		epoch_time_values.append(time.time() - epoch_start)

		# save results to file
		if write_to_file:
			save_results(
				file = os.path.join(reports_path, model.name + '_training.csv'),
				metrics = {
					'id': model.name.upper() + '_' + str(ts),
					'epoch': epoch + 1,
					'model': model.name,
					'train_crossentropy_loss': epoch_loss_train,
					'eval_crossentropy_loss': epoch_loss_eval,
					'exec_time': time.time() - epoch_start,
					'auc_score': metric,
					'datetime': get_date_time()
				}
			)

		# early stopping
		if epoch + 1 - best_metric_epoch == early_stopping:
			print(f"\nEarly stopping triggered at epoch: {str(epoch + 1)}\n")
			break

	print(f"\n\nTrain completed! Best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {str(datetime.timedelta(seconds=int(time.time() - total_start)))}.")
	log.write('['+get_date_time()+'] Training phase ended.EXECUTING: ' + model.name + '\n')
	log.flush()
	log.close()
	return [
		epoch_loss_values[0],
		epoch_loss_values[1],
		epoch_time_values,
		metric_values
	]


def predict_model(
	model,
	data,
	transforms,
	device,
	paths,
	ministep=4,
	num_workers=4,
	write_to_file=True,
	verbose=False
):
	"""
	Standard Pytorch-style prediction program.
	Args:
		model (torch.nn.Module): the model to be loaded.
		data (list): the testing data.
		transform (list): pre transformations for testing data.
		device (str): device's name.
		paths (list): folders where to save results and load model's dump.
		num_workers (int): setting multi-process data loading.
		ministep (int): number of interval of data to load on RAM.
		write_to_file (bool): whether to write results to csv file.
		save_sample (bool): whether to save predicted image samples.
		verbose (bool): whether or not print information.
	Returns:
		metrics (list): dice score and Hausdorff distance for each class.
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	counter = 0
	ministep = ministep if (len(data) > 5 and ministep > 1) else 2
	ministeps_test = np.linspace(0, len(data), ministep).astype(int)
	eval_transform = transforms
	saved_path, reports_path, logs_path = paths

	# log the current execution
	log = open(os.path.join(logs_path, 'prediction.log'), 'a', encoding='utf-8')
	log.write('['+get_date_time()+'] Predictions started.EXECUTING: ' + model.name + '\n')
	log.flush()

	try:
		# load pretrained model
		model.load_state_dict(
			torch.load(os.path.join(saved_path, model.name + '_best.pth'), map_location=torch.device(device))
		)
		model.eval()
		# making inference
		with torch.no_grad():
			y_true, y_pred = [], []
			for i in range(len(ministeps_test) - 1):
				test_ds = CacheDataset(data[ministeps_test[i]:ministeps_test[i+1]], transform=eval_transform, cache_rate=1.0, num_workers=None, progress=False)
				test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
				for val_data in test_loader:
					val_inputs_img, val_inputs_data, val_label = (
						val_data['image'].to(device),
						val_data['data'].to(device),
						val_data['label'].to(device)
					)
					val_outputs = model([val_inputs_img, val_inputs_data])
					y_true.append(val_label.detach().cpu().item())
					y_pred.append(val_outputs.argmax(dim=1).detach().cpu().item())
					counter += 1
					if verbose and ((counter - 1) == 0 or (counter % 20) == 0):
						print(f"inference {counter}/{len(data)}")
						log.write('['+get_date_time()+'] EXECUTING.'+model.name+' INFERENCE '+str(counter)+' OF '+str(len(data))+' \n')
						log.flush()

			# computing metrics
			l = [float(i) for i in range(model.out_channels)]
			matrix = confusion_matrix(y_true, y_pred)
			a = matrix.diagonal() / matrix.sum(axis = 1)
			p = precision_score(y_true, y_pred, labels = l, average = None)
			r = recall_score(y_true, y_pred, labels = l, average = None)
			f = f1_score(y_true, y_pred, labels = l, average = None)
			ra = _roc_auc_score_multiclass(y_true, y_pred)
			a, p, r, f, ra = list(a), list(p), list(r), list(f), list(ra.values())

			# save results to file
			if write_to_file:
				save_results(
					file = os.path.join(reports_path, 'results.csv'),
					metrics = {
						'model': model.name,
						'n_images': len(data),
						'accuracy_score': sum(a) / len(a),
						'accuracy_score_0': a[0],
						'accuracy_score_1': a[1],
						'accuracy_score_2': a[2] if model.out_channels == 3 else '',
						'precision_score': sum(p) / len(p),
						'precision_score_0': p[0],
						'precision_score_1': p[1],
						'precision_score_2': p[2] if model.out_channels == 3 else '',
						'recall_score': sum(r) / len(r),
						'recall_score_0': r[0],
						'recall_score_1': r[1],
						'recall_score_2': r[2] if model.out_channels == 3 else '',
						'f1_score': sum(f) / len(f),
						'f1_score_0': f[0],
						'f1_score_1': f[1],
						'f1_score_2': f[2] if model.out_channels == 3 else '',
						'roc_auc_score': sum(ra) / len(ra),
						'roc_auc_score_0': ra[0],
						'roc_auc_score_1': ra[1],
						'roc_auc_score_2': ra[2] if model.out_channels == 3 else '',
						'datetime': get_date_time()
					}
				)
			log.write('['+get_date_time()+'] Predictions ended.EXECUTING: ' + model.name + '\n')
			log.flush()
			log.close()
			return [a, p, r, f, ra]
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: model dump for\033[95m '+model.name+'\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def _roc_auc_score_multiclass(actual, predicted):
	"""
	Compute roc_auc_score for each class by marking the current class as 1 and all other classes as 0.
	Args:
		actual (list): the actual values.
		metrics (list): the predicted values.
	Returns:
		roc_auc_dict (dict): roc_auc_score for each class as key.
	"""
	unique_class = set(actual)
	roc_auc_dict = {}
	for per_class in unique_class:
		other_class = [x for x in unique_class if x != per_class]
		new_actual = [0 if x in other_class else 1 for x in actual]
		new_predicted = [0 if x in other_class else 1 for x in predicted]
		roc_auc = roc_auc_score(new_actual, new_predicted)
		roc_auc_dict[per_class] = roc_auc
	return roc_auc_dict
