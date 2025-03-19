from .Application import Application

def create_app(task, model, log_path, defense='no'):
    return Application(task, model, log_path, defense)