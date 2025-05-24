# models.py
from datetime import datetime
from db import db

class PlaceCache(db.Model):
    __tablename__ = 'place_cache'
    id            = db.Column(db.Integer,   primary_key=True)
    place_id      = db.Column(db.String,    unique=True, nullable=False)
    name          = db.Column(db.String,    nullable=False)
    keywords_json = db.Column(db.Text,      nullable=False)
    updated_at    = db.Column(db.DateTime, default=datetime.utcnow,
                                              onupdate=datetime.utcnow)
