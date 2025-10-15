from sqlalchemy import Column, Integer, String
from .db import Base

class Pokemon(Base):
    __tablename__ = "pokemons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String)
    color = Column(String)
    habitat = Column(String)
    height = Column(Integer)
    weight = Column(Integer)
    description = Column(String)
