# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: aa-env
#     language: python
#     name: python3
# ---

import os
import ftplib

# FTP details
ftp_host = 'ftp.server.url'
ftp_user = 'username'
ftp_password = 'password'

# Connection to FTP server
ftp = ftplib.FTP(ftp_host)
ftp.login(ftp_user, ftp_password)

# Change directory
ftp.cwd('/RSMC_LaReunion')

# +
# List files in directory
ftp.retrlines('LIST')

# Store files list in variable
filenames = ftp.nlst()  

# +
system = 'ANCHA'

# Check if directory exists locally, otherwise create it
os.makedirs(f"./{system}", exist_ok=True)

# Download all files related to one storm
for remote_path in filenames:
    if system in remote_path:
        if 'json' in remote_path:
            with open(f"./{system}/{remote_path}", 'wb') as local_file:
                ftp.retrbinary('RETR ' + remote_path, local_file.write)
                print(f"{remote_path} has been downloaded with success.")
        else:
            try:
                # Change directory if remote_path is a folder
                ftp.cwd(remote_path)
                print(f"{remote_path} is a folder.")

                # List files in this folder
                files = ftp.nlst()

                for filename in files:
                    with open(f"{system}/{filename}", 'wb') as local_file:
                        ftp.retrbinary('RETR ' + filename, local_file.write)
                        print(f"{filename} has been downloaded with success.")

                ftp.cwd("..")        
            except ftplib.error_perm as e:
                print(f"Error : {e}")

# -

ftp.quit()


