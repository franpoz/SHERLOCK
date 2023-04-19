import gc
import logging
import os
import shutil
import smtplib
import time
import socket
from argparse import ArgumentParser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

from sherlockpipe.bayesian_fit.run import run_fit
from sherlockpipe.search.run import run_search, load_from_yaml
from sherlockpipe.stability import stability_args_parse
from sherlockpipe.system_stability.run import run_stability
from sherlockpipe.validate import validation_args_parse
from sherlockpipe.validation.run import run_validate
from sherlockpipe.vetting.run import run_vet


def move_input_to_working(running_file, working_dir):
    filename = os.path.basename(running_file)
    working_file = working_dir + '/' + filename
    shutil.move(running_file, working_file)
    return working_file

def run_script(input_dir, output_dir, working_dir, cpus, pa):
    running_file = None
    working_file = None
    working_target_dir = None
    properties = None
    try:
        paths = sorted(Path(input_dir).iterdir(), key=os.path.getmtime)
        for index, path in enumerate(paths):
            path_str = str(path)
            if path_str.endswith(".yaml") or path_str.endswith('.yml'):
                running_file = path_str
                working_file = move_input_to_working(running_file, working_dir)
                working_target_dir = working_dir + '/' + Path(working_file).stem
                os.mkdir(working_target_dir)
                os.chdir(working_target_dir)
                properties = run_search(working_file, False, None, cpus=cpus)
            elif os.path.isdir(path_str):
                running_file = path_str
                working_target_dir = move_input_to_working(path_str, working_dir)
                vet_file = working_target_dir + '/vet.yaml'
                validate_file = working_target_dir + '/validate.yaml'
                fit_file = working_target_dir + '/fit.yaml'
                plan_file = working_target_dir + '/plan.yaml'
                stability_file = working_target_dir + '/stability.yaml'
                if os.path.exists(vet_file):
                    properties = load_from_yaml(vet_file)
                    candidate = None if 'CANDIDATE' not in properties else properties['CANDIDATE']
                    run_vet(working_file, candidate, properties, cpus)
                if os.path.exists(validate_file):
                    properties = load_from_yaml(validate_file)
                    candidate = None if 'CANDIDATE' not in properties else properties['CANDIDATE']
                    contrast_curve = None if 'CONTRAST_CURVE' not in properties else properties['CONTRAST_CURVE']
                    bins = None if 'BINS' not in properties else properties['BINS']
                    scenarios = None if 'SCENARIOS' not in properties else properties['SCENARIOS']
                    sigma_mode = None if 'SIGMA_MODE' not in properties else properties['SIGMA_MODE']
                    args = validation_args_parse({'object_dir': path_str, 'candidate': candidate,
                                                  'properties': properties, 'contrast_curve': contrast_curve,
                                                  'bins': bins, 'scenarios': scenarios, 'sigma_mode': sigma_mode})
                    run_validate(args)
                if os.path.exists(fit_file):
                    properties = load_from_yaml(fit_file)
                    candidate = None if 'CANDIDATE' not in properties else properties['CANDIDATE']
                    only_initial = None if 'ONLY_INITIAL' not in properties else properties['ONLY_INITIAL']
                    tolerance = None if 'TOLERANCE' not in properties else properties['TOLERANCE']
                    mcmc = None if 'MCMC' not in properties else properties['MCMC']
                    detrend = None if 'DETREND' not in properties else properties['DETREND']
                    fit_orbit = None if 'FIT_ORBIT' not in properties else properties['FIT_ORBIT']
                    args = validation_args_parse({'object_dir': path_str, 'candidate': candidate,
                                                  'properties': properties, 'only_initial': only_initial,
                                                  'tolerance': tolerance, 'mcmc': mcmc, 'detrend': detrend,
                                                  'fit_orbit': fit_orbit})
                    run_fit(args)
                if os.path.exists(plan_file):
                    properties = load_from_yaml(vet_file)
                if os.path.exists(stability_file):
                    properties = load_from_yaml(vet_file)
                    max_ecc = None if 'MAX_ECC' not in properties else properties['MAX_ECC']
                    period_bins = None if 'PERIOD_BINS' not in properties else properties['PERIOD_BINS']
                    ecc_bins = None if 'ECC_BINS' not in properties else properties['ECC_BINS']
                    inc_bins = None if 'INC_BINS' not in properties else properties['INC_BINS']
                    omega_bins = None if 'OMEGA_BINS' not in properties else properties['OMEGA_BINS']
                    mass_bins = None if 'MASS_BINS' not in properties else properties['MASS_BINS']
                    star_mass_bins = None if 'STAR_MASS_BINS' not in properties else properties['STAR_MASS_BINS']
                    years = None if 'YEARS' not in properties else properties['YEARS']
                    spock = None if 'SPOCK' not in properties else properties['SPOCK']
                    free_params = None if 'CANDIDATE' not in properties else properties['CANDIDATE']
                    args = stability_args_parse({'object_dir': path_str, 'max_ecc': max_ecc,
                                                  'period_bins': period_bins, 'ecc_bins': ecc_bins,
                                                  'inc_bins': inc_bins, 'omega_bins': omega_bins,
                                                  'mass_bins': mass_bins, 'star_mass_bins': star_mass_bins,
                                                  'years': years, 'spock': spock,
                                                  'free_params': free_params})
                    run_stability(args)
    finally:
        os.chdir(working_dir)
        if running_file is not None and os.path.exists(running_file):
            shutil.rmtree(running_file, ignore_errors=True)
        if working_file is not None and os.path.exists(working_file):
            shutil.rmtree(working_file, ignore_errors=True)
        if working_target_dir is not None and os.path.exists(working_target_dir):
            logging.info("Finished file %s", running_file)
            output_file = output_dir + '/' + Path(working_target_dir).stem
            shutil.move(working_target_dir, output_file)
            if properties is not None and 'EMAIL' in properties:
                receiver_address = properties['EMAIL']
                send_email(output_file, receiver_address, pa)
            else:
                logging.warning('Cannot send an email because the receiver email was not provided.')
        gc.collect()


def send_email(filename, receiver_address, pa):
    try:
        sender_address = 'sherlockpipeline@gmail.com'
        sender_pass = pa
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        host = socket.gethostname()
        try:
            host = host if host != 'localhost' else requests.get('https://api.ipify.org').content.decode('utf8')
        except:
            print('Cant identify host in the internet')
        message['Subject'] = 'SHERLOCK [' + host + ']: Job with filename ' + os.path.basename(filename) + ' finished'
        # The body and the attachments for the mail
        message.attach(MIMEText('You can check all the target directories under the directory: ' + filename +
                                '. Keep in mind that these files will be kept only for two weeks and then will be '
                                ' automatically removed.', 'plain'))
        # Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
        session.starttls()  # enable security
        session.login(sender_address, sender_pass)  # login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        logging.info('Sent email to %s for job %s', receiver_address, filename)
    except Exception:
        logging.exception('Cant send email to %s for job %s', receiver_address, filename)


def clean_old_outputs(output_dir):
    current_time = time.time()
    max_days = 14
    files = os.listdir(output_dir)
    for file in files:
        file_path = os.path.join(output_dir, file)
        file_mod_time = os.path.getmtime(file_path)
        time_diff = current_time - file_mod_time
        if time_diff > max_days * 24 * 60 * 60:
            shutil.rmtree(file_path, ignore_errors=True)


if __name__ == '__main__':
    ap = ArgumentParser(description='Sherlock As A Service')
    ap.add_argument('--input_dir',
                    help="Drop-in directory for users",
                    required=True, type=str)
    ap.add_argument('--working_dir',
                    help="Directory to be used when working with a file",
                    required=True, type=str)
    ap.add_argument('--output_dir',
                    help="Directory where SHERLOCK will drop the results",
                    required=True, type=str)
    ap.add_argument('--cpus',
                    help="Number of cpu processes to be used",
                    required=True, type=int)
    ap.add_argument('--pa',
                    help="",
                    required=True, type=str)
    args = ap.parse_args()
    paths = sorted(Path(args.working_dir).iterdir(), key=os.path.getmtime)
    for index, path in enumerate(paths):
        file_in_working = str(path)
        if file_in_working.endswith(".yaml") or file_in_working.endswith('.yml'):
            filename = os.path.basename(file_in_working)
            shutil.move(file_in_working, args.input_dir + '/' + filename)
        else:
            shutil.rmtree(file_in_working, ignore_errors=True)
    if not isinstance(logging.root, logging.RootLogger):
        logging.root = logging.RootLogger(logging.INFO)
    from apscheduler.schedulers.blocking import BlockingScheduler
    logging.getLogger('apscheduler.executors.default').setLevel(logging.ERROR)
    executors = {
        'default': ProcessPoolExecutor(1),
        'processpool': ProcessPoolExecutor(1)
    }
    scheduler = BlockingScheduler(executors)
    scheduler.add_job(run_script, 'interval', minutes=1, max_instances=1, misfire_grace_time=600000,
                      kwargs={"input_dir": args.input_dir, "output_dir": args.output_dir,
                              "working_dir": args.working_dir, "cpus": args.cpus, "pa": args.pa}, replace_existing=True)
    scheduler.add_job(clean_old_outputs, 'interval', days=1, max_instances=1,
                      kwargs={"output_dir": args.output_dir}, replace_existing=True)
    scheduler.start()
