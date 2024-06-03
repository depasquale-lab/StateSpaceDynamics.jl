# Documentor.jl Workflow Quickstart

## Initial Set-Up (one time)
Activate the Julia project environment found at ".../ssm_julia/docs". 

Add your local version of the SSM package to environment. In the package manager, enter the command ```dev <your_path_to_local_repo>/ssm_julia```. This adds your local version of the SSM package to the "docs" project environment. Next, run the command ```status``` in the package manager. Ensure the outputted packages include:
Documenter
LiveServer
Revise
SSM

If these packages are included, your project environment is all set!

## Work Session Set-Up

Run "make.jl". This file automatically rebuilds the documentation each time a change is made to the SSM package (or docs/src directory). 

Using a new process (or REPL), run "server.jl". With any browswer, open the localhost URL displayed. This script runs a local server to view the built documentation (this is the officially recommended method for viewing). 

Leave both processes running throughout your work session. You are good to go!

## Recommended Reading

Learn how to work with Documentor.jl here: https://documenter.juliadocs.org/stable/man/guide/.