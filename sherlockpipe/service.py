import logging
import os
import shutil
import smtplib
import time
from argparse import ArgumentParser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from sherlockpipe.loading.run import run

file_in_working = ''

def run_script(input_dir, output_dir, working_dir, cpus, pa):
    running_file = None
    working_file = None
    working_target_dir = None
    user_properties = None
    try:
        paths = sorted(Path(input_dir).iterdir(), key=os.path.getmtime)
        for index, path in enumerate(paths):
            if str(path).endswith(".yaml") or str(path).endswith('.yml'):
                running_file = str(path)
                filename = os.path.basename(running_file)
                working_file = working_dir + '/' + filename
                shutil.move(running_file, working_file)
                working_target_dir = working_dir + '/' + Path(working_file).stem
                os.mkdir(working_target_dir)
                os.chdir(working_target_dir)
                user_properties = run(working_file, False, cpus)
    finally:
        os.chdir(working_dir)
        if running_file is not None and os.path.exists(running_file):
            os.remove(running_file)
        if working_file is not None and os.path.exists(working_file):
            os.remove(working_file)
        if working_target_dir is not None and os.path.exists(working_file):
            logging.info("Finished file %s", running_file)
            output_file = output_dir + Path(working_target_dir).stem
            shutil.move(working_target_dir, output_file)
            if user_properties is not None and 'EMAIL' in user_properties:
                receiver_address = user_properties['EMAIL']
                send_email(output_file, receiver_address, pa)
            else:
                logging.warning('Cannot send an email because the receiver email was not provided.')
        else:
            logging.info("No pending tasks found")


def send_email(filename, receiver_address, pa):
    try:
        sender_address = 'sherlockpipeline@gmail.com'
        sender_pass = pa
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = 'SHERLOCK: Job with filename ' + os.path.basename(filename) + ' finished'
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
    send_email("/home/blabla", 'martin.devora.pajares@gmail.com')
    paths = sorted(Path(args.working_dir).iterdir(), key=os.path.getmtime)
    for index, path in enumerate(paths):
        file_in_working = str(path)
        if file_in_working.endswith(".yaml") or file_in_working.endswith('.yml'):
            filename = os.path.basename(file_in_working)
            shutil.move(file_in_working, args.input_dir + '/' + filename)
        else:
            shutil.rmtree(file_in_working, ignore_errors=True)
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(run_script, 'interval', minutes=1, max_instances=1, misfire_grace_time=600000,
                      kwargs={"input_dir": args.input_dir, "output_dir": args.output_dir,
                              "working_dir": args.working_dir, "cpus": args.cpus, "pa": args.pa})
    scheduler.add_job(clean_old_outputs, 'interval', days=1, max_instances=1,
                      kwargs={"output_dir": args.output_dir})
    scheduler.start()
