from flask import Flask
from config import Config
from flasgger import Swagger


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    Swagger(app, template_file='apidocs/main.yaml', merge=True, config={'specs_route': '/api-docs/'})
    
    from . import home
    app.register_blueprint(home.bp)
    
    from . import sfr
    app.register_blueprint(sfr.bp)
    
    return app
