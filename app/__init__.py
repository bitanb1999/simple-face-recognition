from flask import Flask
from config import Config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    
    @app.route('/foo')
    def serve_foo():
        return 'This page is served via Flask!'
    
    from . import home
    app.register_blueprint(home.bp)
    
    from . import sfr
    app.register_blueprint(sfr.bp)
    
    return app
