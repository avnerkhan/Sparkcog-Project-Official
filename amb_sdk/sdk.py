import requests
import urllib
import time
import os.path
import tempfile
import validators
import json
import zipfile
import io
import pandas as pd
from requests_toolbelt.multipart import encoder
from amb_sdk.config import Config as cfg


class DarwinSdk:

    auth_string = ''
    api_key = ''
    password = ''
    username = ''
    user_password = ''
    token_start_time = 0
    token_time_limit = 3500
    cfg = cfg

    s = requests.Session()
    server_url = cfg.server_url
    version = 'v1'
    routes = {'auth_login': 'auth/login',
              'auth_login_user': 'auth/login/user',
              'auth_register': 'auth/register',
              'auth_register_user': 'auth/register/user',
              'auth_change_password': 'auth/password',
              'auth_reset_password': 'auth/password/reset',
              'auth_set_email': 'auth/email',
              'auth_delete_user': 'auth/register/user/',
              'lookup_job_status': 'job/status',
              'lookup_job_status_name': 'job/status/',
              'delete_job':  'job/status/',
              'stop_job':  'job/status/',
              'lookup_artifact': 'lookup/artifact',
              'lookup_artifact_name': 'lookup/artifact/',
              'lookup_limits': 'lookup/limits',
              'lookup_dataset': 'lookup/dataset',
              'lookup_dataset_name': 'lookup/dataset/',
              'lookup_model': 'lookup/model',
              'lookup_model_name': 'lookup/model/',
              'lookup_tier': 'lookup/tier',
              'lookup_tier_num': 'lookup/tier/',
              'lookup_user': 'lookup/user',
              'lookup_username': 'lookup/user/',
              'display_population': 'lookup/model/{}/population',
              'get_info': 'info',
              'create_model': 'train/model',
              'delete_model': 'train/model/',
              'resume_training_model': 'train/model/',
              'upload_dataset': 'upload/',
              'delete_dataset': 'upload/',
              'download_artifact': 'download/artifacts/',
              'download_dataset': 'download/dataset/',
              'download_model': 'download/model/',
              'delete_artifact': 'download/artifacts/',
              'analyze_data': 'analyze/data/',
              'analyze_model': 'analyze/model/',
              'analyze_predictions': 'analyze/model/predictions/',
              'clean_data': 'clean/dataset/',
              'create_risk_info': 'risk/',
              'run_model': 'run/model/',
              'set_url': '',
              'get_url': '',
              'delete_all_datasets': '',
              'delete_all_models': '',
              'delete_all_artifacts': '',
              'wait_for_job': ''}

    # Set URL
    def set_url(self, url, version='v1'):
        if validators.url(url):
            self.server_url = url
            self.version = version
            return True, self.server_url
        else:
            return False, 'invalid url'

    def get_url(self):
        return True, self.server_url

    # Authentication and Registration
    def auth_login(self, password, api_key):
        self.username = ''
        self.api_key = api_key
        self.password = password
        url = self.server_url + self.routes['auth_login']
        payload = {'api_key': str(api_key), 'pass1': str(password)}
        r = self.s.post(url, data=payload)
        # r = self.s.post(url, data=payload, verify=False)
        if r.ok:
            self.auth_string = 'Bearer ' + r.json()['access_token']
            self.token_start_time = time.time()
            return True, self.auth_string
        else:
            return False, '{}: {} - {}'.format(r.status_code, r.reason, r.text[0:1024])

    def auth_login_user(self, username, password):
        self.username = username
        self.password = password
        url = self.server_url + self.routes['auth_login_user']
        payload = {'username': str(username), 'pass1': str(password)}
        r = self.s.post(url, data=payload)
        # r = self.s.post(url, data=payload, verify=False)
        if r.ok:
            self.auth_string = 'Bearer ' + r.json()['access_token']
            self.token_start_time = time.time()
            return True, self.auth_string
        else:
            return False, '{}: {} - {}'.format(r.status_code, r.reason, r.text[0:1024])

    def auth_register(self, password, api_key, email):
        self.username = ''
        self.password = password
        self.api_key = api_key
        url = self.server_url + self.routes['auth_register']
        headers = {'Authorization': self.auth_string}
        payload = {'api_key': str(api_key), 'pass1': str(password), 'pass2': str(password), 'email': str(email)}
        r = self.s.post(url, headers=headers, data=payload)
        if r.ok:
            self.auth_string = 'Bearer ' + r.json()['access_token']
            self.token_start_time = time.time()
            return True, self.auth_string
        else:
            return False, '{}: {} - {}'.format(r.status_code, r.reason, r.text[0:1024])

    def auth_register_user(self, username, password, email):
        self.username = username
        self.password = password
        url = self.server_url + self.routes['auth_register_user']
        headers = {'Authorization': self.auth_string}
        payload = {'username': str(username), 'pass1': str(password), 'pass2': str(password), 'email': str(email)}
        r = self.s.post(url, headers=headers, data=payload)
        if r.ok:
            self.auth_string = 'Bearer ' + r.json()['access_token']
            self.token_start_time = time.time()
            return True, self.auth_string
        else:
            return False, '{}: {} - {}'.format(r.status_code, r.reason, r.text[0:1024])

    def auth_change_password(self, curpass, newpass):
        url = self.server_url + self.routes['auth_change_password']
        headers = {'Authorization': self.auth_string}
        payload = {'curpass': str(curpass), 'newpass1': str(newpass), 'newpass2': str(newpass)}
        r = self.s.patch(url, headers=headers, data=payload)
        if r.ok:
            self.password = newpass
            return True, None
        else:
            return False, '{}: {} - {}'.format(r.status_code, r.reason, r.text[0:1024])

    def auth_reset_password(self, username):
        url = self.server_url + self.routes['auth_reset_password']
        headers = self.get_auth_header()
        payload = {'username': str(username)}
        r = self.s.patch(url, headers=headers, data=payload)
        return self.get_return_info(r)

    def auth_set_email(self, username, email):
        url = self.server_url + self.routes['auth_set_email']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        payload = {'username': username, 'email': email}
        r = self.s.patch(url, headers=headers, data=payload)
        return self.get_return_info(r)

    def auth_delete_user(self, username):
        url = self.server_url + self.routes['auth_delete_user'] + urllib.parse.quote(username, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.delete(url, headers=headers)
        return self.get_return_info(r)

    # Conveniences
    def get_auth_header(self):
        if not self.username and not self.api_key:
            # Either api_key or username must be set.
            return None
        if time.time() - self.token_start_time > self.token_time_limit:
            if self.username:
                self.auth_login_user(self.username, self.password)
            else:
                self.auth_login(self.password, self.api_key)
        return {'Authorization': self.auth_string}

    def get_return_info(self, r):
        if r.ok:
            if not r.text:
                return True, None
            else:
                return True, r.json()
        else:
            return False, '{}: {} - {}'.format(r.status_code, r.reason, r.text[0:1024])

    # Job methods
    def lookup_job_status(self, age=None, status=None):
        url = self.server_url + self.routes['lookup_job_status']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        payload = {'age': age, 'status': status}
        r = self.s.get(url, headers=headers, params=payload)
        return self.get_return_info(r)

    def lookup_job_status_name(self, job_name):
        url = self.server_url + self.routes['lookup_job_status_name']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url + urllib.parse.quote(job_name, safe=''), headers=headers)
        return self.get_return_info(r)

    def delete_job(self, job_name):
        url = self.server_url + self.routes['delete_job']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.delete(url + urllib.parse.quote(job_name, safe=''), headers=headers)
        return self.get_return_info(r)

    def stop_job(self, job_name):
        url = self.server_url + self.routes['stop_job']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.patch(url + urllib.parse.quote(job_name, safe=''), headers=headers)
        return self.get_return_info(r)

    # Get model or dataset metadata
    def lookup_artifact(self, type=None):
        url = self.server_url + self.routes['lookup_artifact']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        payload = {'type': type}
        r = self.s.get(url, headers=headers, params=payload)
        return self.get_return_info(r)

    def lookup_artifact_name(self, artifact_name):
        url = self.server_url + self.routes['lookup_artifact_name']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url + urllib.parse.quote(artifact_name, safe=''), headers=headers)
        return self.get_return_info(r)

    def lookup_limits(self):
        url = self.server_url + self.routes['lookup_limits']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_dataset(self):
        url = self.server_url + self.routes['lookup_dataset']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_dataset_name(self, dataset_name):
        url = self.server_url + self.routes['lookup_dataset_name']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url + urllib.parse.quote(dataset_name, safe=''), headers=headers)
        return self.get_return_info(r)

    def lookup_model(self):
        url = self.server_url + self.routes['lookup_model']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_model_name(self, model_name):
        url = self.server_url + self.routes['lookup_model_name'] + urllib.parse.quote(model_name, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_tier(self):
        url = self.server_url + self.routes['lookup_tier']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_tier_num(self, tier_num):
        url = self.server_url + self.routes['lookup_tier_num'] + urllib.parse.quote(str(tier_num), safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_user(self):
        url = self.server_url + self.routes['lookup_user']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def lookup_username(self, username):
        url = self.server_url + self.routes['lookup_username'] + urllib.parse.quote(username, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def get_info(self):
        url = self.server_url + self.routes['get_info']
        r = self.s.get(url)
        return self.get_return_info(r)

    # Train a model
    def create_model(self, dataset_names, **kwargs):
        url = self.server_url + self.routes['create_model']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        parameters = kwargs
        if 'dataset_names' not in parameters:
            if isinstance(dataset_names, str):
                parameters['dataset_names'] = [dataset_names]
            else:
                parameters['dataset_names'] = dataset_names
        r = self.s.post(url, headers=headers, json=parameters)
        return self.get_return_info(r)

    def delete_model(self, model_name):
        url = self.server_url + self.routes['delete_model'] + urllib.parse.quote(model_name, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.delete(url, headers=headers)
        return self.get_return_info(r)

    def resume_training_model(self, model_name, dataset_names, **kwargs):
        url = self.server_url + self.routes['resume_training_model'] + urllib.parse.quote(model_name, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        parameters = kwargs
        if 'dataset_names' not in parameters:
            if isinstance(dataset_names, str):
                parameters['dataset_names'] = [dataset_names]
            else:
                parameters['dataset_names'] = dataset_names
        r = self.s.patch(url, headers=headers, json=parameters)
        return self.get_return_info(r)

    # Upload or delete a dataset
    def upload_dataset(self, dataset_path, dataset_name=None, has_header=True):
        if dataset_name is None:
            head, tail = os.path.split(dataset_path)
            dataset_name = tail
            # dataset_name = dataset_path.split('/')[-1]
        url = self.server_url + self.routes['upload_dataset']
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        if not os.path.isfile(dataset_path):
            return False, "File not found"
        with open(dataset_path, 'rb') as f:
            form = encoder.MultipartEncoder({
                "dataset": (str(dataset_path), f, 'text/csv/h5'),
                'dataset_name': str(dataset_name),
                'has_header': str(has_header)
            })
            headers.update({"Prefer": "respond-async", "Content-Type": form.content_type})
            r = self.s.post(url, headers=headers, data=form)
        return self.get_return_info(r)

    def delete_dataset(self, dataset_name):
        url = self.server_url + self.routes['delete_dataset'] + urllib.parse.quote(dataset_name, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.delete(url, headers=headers)
        return self.get_return_info(r)

    # Upload or delete a generated artifact
    def download_artifact(self, artifact_name, artifact_path=None):
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."

        if artifact_path:
            (code, response) = self._validate_artifact_file_path(artifact_path)
            if not code:
                return False, response

        artifact_type = None
        (code, response) = self.lookup_artifact_name(artifact_name)
        if code is True:
            artifact_type = response['type']
        url = self.server_url + self.routes['download_artifact'] + urllib.parse.quote(artifact_name, safe='')

        r = self.s.get(url, headers=headers)
        (code, response) = self.get_return_info(r)
        if code is True:
            artifact = response['artifact']
            if artifact_type == 'Model':
                if artifact[1:4] == 'PNG':
                    return self._write_to_file(artifact_path, '.png', artifact)
                else:
                    data = json.loads(artifact)
                    if 'global_feat_imp' in data:
                        df = pd.Series(data['global_feat_imp']).sort_values(ascending=False)
                    elif 'local_feat_imp' in data:
                        df = pd.DataFrame(data['local_feat_imp'])
                        df.index = df.index.astype(int)
                        df = df.sort_index()
                    else:
                        return False, "Unknown artifact format for model"
                    return True, df
            if artifact_type == 'Test':
                data = json.loads(response['artifact'])
                if 'index' in data:
                    if len(data["index"]) == len(data['actual']):
                        df = pd.DataFrame({'index': data['index'], 'actual': data['actual'],
                                           'predicted': data['predicted']})
                    else:
                        df = pd.DataFrame({'actual': data['actual'], 'predicted': data['predicted']})
                    return True, df
                elif 'x' in data:
                    if len(data["x"]) == len(data['actual']):
                        df = pd.DataFrame({'index': data['x'], 'actual': data['actual'],
                                           'predicted': data['predicted']})
                    else:
                        df = pd.DataFrame({'actual': data['actual'], 'predicted': data['predicted']})
                    return True, df
                else:
                    return False, "Cannot interpret Test artifact"
            if artifact_type == 'Risk':
                return self._write_to_file(artifact_path, '.csv', artifact)

            if artifact_type == 'Run':
                if 'png' in artifact[0:100]:
                    return self._write_to_file(artifact_path, '.zip', artifact)

                if 'anomaly' in artifact[0:50]:
                    return self._write_to_file(artifact_path, '.csv', artifact)

                if DarwinSdk.is_json(response['artifact']):
                    data = json.loads(response['artifact'])
                    if 'index' in data:
                        if len(data["index"]) == len(data['actual']):
                            df = pd.DataFrame({'index': data['index'], 'actual': data['actual'],
                                               'predicted': data['predicted']})
                            return True, df
                        else:
                            df = pd.DataFrame({'actual': data['actual'], 'predicted': data['predicted']})
                            return True, df
                    else:
                        df = pd.read_json(json.dumps(data), orient='records')
                        if artifact_path:
                            csv_path = os.path.join(artifact_path, 'artifact.csv')
                            df.to_csv(csv_path, encoding='utf-8', index=False)
                            return True, {"filename": csv_path}
                        else:
                            return True, df
                else:
                    data = response['artifact'].splitlines()
                    col_name = data[0]
                    del data[0]
                    df = pd.DataFrame(data, columns=[col_name])
                    if DarwinSdk.is_number(df[col_name][0]):
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    return True, df
            if artifact_type in ['Dataset', 'CleanDataTiny']:
                data = json.loads(response['artifact'])
                df = pd.DataFrame(data=data[0], index=[0])
                for x in range(1, len(data)):
                    df = df.append(data[x], ignore_index=True)
                return True, df
            if self._is_local() and artifact_type in ('AnalyzeData'):
                # for onprem, we have to intepret artifact differently
                data = json.loads(response['artifact'])
                df = pd.DataFrame(data=data[0], index=[0])
                for x in range(1, len(data)):
                    df = df.append(data[x], ignore_index=True)
                return True, df
            if artifact_type in ['AnalyzeData', 'CleanData']:
                buf = '[' + response['artifact'] + ']'
                buf = buf.replace('}', '},').replace('\n', '').replace(',]', ']')
                data = json.loads(buf)
                df = pd.DataFrame(data=data[0], index=[0])
                for x in range(1, len(data)):
                    df = df.append(data[x], ignore_index=True)
                return True, df
            if artifact_type in ['CleanDataTiny']:
                df = pd.read_csv(io.StringIO(response['artifact']), sep=",")
                return True, df
            return False, "Unknown artifact type"
        else:
            return False, response

    # Download a dataset (artifact or original dataset)
    def download_dataset(self, dataset_name, file_part=None, artifact_path=None):
        artifact_name = dataset_name
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."

        if artifact_path:
            (code, response) = self._validate_artifact_file_path(artifact_path)
            if not code:
                return False, response

        (code, response) = self.lookup_artifact_name(artifact_name)
        if code is True:  # artifact dataset
            artifact_type = response['type']
            url = self.server_url + self.routes['download_dataset'] + urllib.parse.quote(artifact_name, safe='')
            r = self.s.get(url, headers=headers)
            (code, response) = self.get_return_info(r)
            if code is True:
                artifact = response['dataset']
                if artifact_type in ['CleanData', 'CleanDataTiny']:
                    file_prefix = dataset_name + '-cleaned-'
                    return self._write_to_file(artifact_path, '.csv', artifact, prefix=file_prefix)
            return False, "Unknown dataset artifact type"
        else:  # original dataset
            (code, response) = self.lookup_dataset_name(artifact_name)
            if code is True:
                payload = {'file_part': file_part}
                url = self.server_url + self.routes['download_dataset'] + urllib.parse.quote(artifact_name, safe='')
                r = self.s.get(url, headers=headers, data=payload)
                (code, response) = self.get_return_info(r)
                if code is True:
                    dataset = response['dataset']
                    part = response['part']
                    note = response['note']
                    file_prefix = dataset_name + '-part' + str(part) + '-'
                    response = self._write_to_file(artifact_path, '.csv', dataset, prefix=file_prefix)
                    response[1]['part'] = part
                    response[1]['note'] = note
                    return response
                return False, "Unknown dataset"
            else:
                return False, response

    def delete_artifact(self, artifact_name):
        url = self.server_url + self.routes['delete_artifact'] + urllib.parse.quote(artifact_name, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.delete(url, headers=headers)
        return self.get_return_info(r)

    def download_model(self, model_name, path=None, model_type=None, model_format=None):
        """
        Download a model and data profiler given a model_name and location
        If location is not supplied, it will download to the current directory
        :param model_name: Model name to download
        :param path: Path where the model and data profiler are supposed to be downloaded
        :param model_type: Model type of the model
        :param model_format: Format of the model to be downloaded
        :return: Response if the download was successful or not
        """
        headers = {'Authorization': self.auth_string}
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        url = self.server_url + self.routes['download_model'] + urllib.parse.quote(model_name, safe='')
        payload = {}
        if model_type:
            payload['model_type'] = model_type
        if model_format:
            payload['model_format'] = model_format
        r = self.s.get(url, headers=headers, stream=True, params=payload)
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=path)
        except:
            return False, "Error while downloading models"
        return True, None

    # Analyze a model or data set
    def analyze_data(self, dataset_name, **kwargs):
        url = self.server_url + self.routes['analyze_data'] + urllib.parse.quote(dataset_name, safe='')
        headers = self.get_auth_header()
        parameters = kwargs
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.post(url, headers=headers, json=parameters)
        return self.get_return_info(r)

    # Analyze global feature importances
    def analyze_model(self, model_name, job_name=None, artifact_name=None, category_name=None, model_type=None):
        url = self.server_url + self.routes['analyze_model'] + urllib.parse.quote(model_name, safe='')
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        payload = {'job_name': job_name, 'artifact_name': artifact_name,
                   'category_name': category_name, 'model_type': model_type}
        r = self.s.post(url, headers=headers, data=payload)
        return self.get_return_info(r)

    # Analyze sample-wise feature importances
    def analyze_predictions(self, model_name, dataset_name, job_name=None, artifact_name=None, model_type=None):
        url = self.server_url + self.routes['analyze_predictions'] + str(model_name) + '/' + dataset_name
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        payload = {'job_name': job_name, 'artifact_name': artifact_name, 'model_type': model_type}
        r = self.s.post(url, headers=headers, data=payload)
        return self.get_return_info(r)

    # Clean a data set
    def clean_data(self, dataset_name, **kwargs):
        url = self.server_url + self.routes['clean_data'] + urllib.parse.quote(dataset_name, safe='')
        headers = self.get_auth_header()
        parameters = kwargs
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        r = self.s.post(url, headers=headers, json=parameters)
        return self.get_return_info(r)

    # Create risk information for a datatset
    def create_risk_info(self, failure_data, timeseries_data, job_name=None, artifact_name=None, **kwargs):
        url = self.server_url + self.routes['create_risk_info'] + failure_data + '/' + timeseries_data
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        parameters = kwargs
        if 'job_name' not in parameters and job_name is not None:
            parameters['job_name'] = job_name
        if 'artifact_name' not in parameters and artifact_name is not None:
            parameters['artifact_name'] = artifact_name
        r = self.s.post(url, headers=headers, json=parameters)
        return self.get_return_info(r)

    # Run a model on some dataset
    def run_model(self, dataset_name, model_name, **kwargs):
        url = self.server_url + self.routes['run_model'] + model_name + '/' + dataset_name
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        parameters = kwargs
        r = self.s.post(url, headers=headers, json=parameters)
        return self.get_return_info(r)

    # Interactive help
    def help(self):
        import inspect
        for key, value in sorted(DarwinSdk.routes.items()):
            print("   ", key, inspect.signature(getattr(DarwinSdk, str(key))))

    # User convenience
    def delete_all_models(self):
        (code, response) = self.lookup_model()
        if code:
            for model in response:
                model_name = model['name']
                print('Deleting {}'.format(model_name))
                (c, r) = self.delete_model(model_name)
                if not c:
                    print('Error removing model "{}" - {}'.format(model_name, r))
            return True, None
        else:
            return False, None

    def delete_all_datasets(self):
        (code, response) = self.lookup_dataset()
        if code:
            for dataset in response:
                dataset_name = dataset['name']
                print('Deleting {}'.format(dataset_name))
                (c, r) = self.delete_dataset(dataset_name)
                if not c:
                    print('Error removing dataset "{}" - {}'.format(dataset_name, r))
            return True, None
        else:
            return False, None

    def delete_all_artifacts(self):
        (code, response) = self.lookup_artifact()
        if code:
            for artifact in response:
                artifact_name = artifact['name']
                print('Deleting {}'.format(artifact_name))
                (c, r) = self.delete_artifact(artifact_name)
                if not c:
                    print('Error removing artifact "{}" - {}'.format(artifact_name, r))
            return True, None
        else:
            return False, None

    def wait_for_job(self, job_name, time_limit=600):
        start_time = time.time()
        (code, response) = self.lookup_job_status_name(str(job_name))
        print(response)
        if type(response) is dict:
            while (response['percent_complete'] != 100):
                if (time.time() - start_time > time_limit):
                    break
                time.sleep(15.0)
                (code, response) = self.lookup_job_status_name(str(job_name))
                print(response)
                if type(response) is not dict:
                    return False, response
            if response['percent_complete'] < 100:
                return(False, "Waited for " + str(time_limit / 60) +
                       " minutes. Re-run wait_for_job to wait longer.")
            if response['percent_complete'] == 100 and response['status'] != 'Failed':
                return (True, "Job completed")
            return False, response
        else:
            return False, response

    def display_population(self, model_name):
        """
        Display population for the given model name
        :param model_name: model name for which the population is to be displayed
        :return: Json string with the population display
        """
        headers = self.get_auth_header()
        if headers is None:
            return False, "Cannot get Auth token. Please log in."
        url = self.server_url + self.routes['display_population'].format(model_name)
        r = self.s.get(url, headers=headers)
        return self.get_return_info(r)

    def _validate_artifact_file_path(self, artifact_path):

        if not os.path.isdir(artifact_path):
            return False, "Invalid Directory or Path"

        if not os.access(artifact_path, os.W_OK):
            return False, "Directory does not have write permissions"

        return True, ""

    def _write_to_file(self, artifact_path, suffix, artifact, prefix='artifact-'):
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False, dir=artifact_path) as file:
            filename = file.name
            file.write(artifact.encode('latin-1'))
            return True, {"filename": filename}

    def _is_local(self):
        c, r = self.get_info()
        return r['local']

    # private
    @staticmethod
    def is_json(myjson):
        try:
            json.loads(myjson)
        except ValueError as e:
            return False
        return True

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
