# Documentor.jl Workflow Quickstart

## Initial Set-Up (one time)
Activate the Julia project environment found at "<path to repo>/ssm_julia/docs" (use control shift P in vscode, then search activate julia environment).

Add your local version of the SSM package to environment. In the package manager, enter the command ```dev <your_path_to_local_repo>/ssm_julia```. This adds your local version of the SSM package to the "docs" project environment. Next, run the command ```status``` in the package manager. Ensure the outputted packages include:
Documenter
LiveServer
Revise
SSM

If these packages are included, your project environment is all set!

## Work Session Set-Up
If using VSCode: It is recommended to use a separate window for documentation development as opposed to package development.


Run "make.jl". This file automatically rebuilds the documentation each time a change is made to the SSM package (or docs/src directory). After the line "Info: Automatic `version="0.0.0"` for inventory from ..\Project.toml", the build is complete. Leave this file running, as it will rebuild in real time when you change docstrings in src/.


Using a new process (or REPL), run "server.jl". VSCode sometimes doesn't let you run the second process in the same window, so just create a new window (control shift P, then search duplicate window) and set the julia project envirnoment as above. With any browswer, open the localhost URL displayed. This script runs a local server to view the built documentation (this is the officially recommended method for viewing, according Documentor.jl docs).


Leave both processes running throughout your work session. You are good to go!

## Recommended Reading

Learn how to work with Documenter.jl here: https://documenter.juliadocs.org/stable/man/guide/.