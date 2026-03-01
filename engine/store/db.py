import logging
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Date, UniqueConstraint

Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, unique=True, nullable=False) # Source ID (e.g. understat ID)
    date_utc = Column(DateTime, nullable=False)
    competition = Column(String, nullable=False) # e.g., EPL, LALIGA
    season = Column(Integer, nullable=False) # e.g., 2023 for 23/24 season
    
    home_team_id = Column(String, nullable=False)
    away_team_id = Column(String, nullable=False)
    home_team_name = Column(String, nullable=False)
    away_team_name = Column(String, nullable=False)
    
    home_goals_ft = Column(Integer, nullable=False)
    away_goals_ft = Column(Integer, nullable=False)
    
    # Optional stats (Understat)
    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)
    home_shots = Column(Integer, nullable=True)
    away_shots = Column(Integer, nullable=True)
    
    # Enhanced Stats (API-Football)
    apifootball_id = Column(Integer, unique=True, nullable=True)
    home_possession = Column(Integer, nullable=True)
    away_possession = Column(Integer, nullable=True)
    home_sot = Column(Integer, nullable=True)
    away_sot = Column(Integer, nullable=True)
    home_lineup = Column(String, nullable=True) # Stored as JSON string
    away_lineup = Column(String, nullable=True) # Stored as JSON string
    goal_events = Column(String, nullable=True) # Stored as JSON string of minutes and teams
    referee = Column(String, nullable=True)
    odds_1x2_home = Column(Float, nullable=True)
    odds_1x2_draw = Column(Float, nullable=True)
    odds_1x2_away = Column(Float, nullable=True)
    
    is_played = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint('date_utc', 'competition', 'home_team_id', 'away_team_id', name='uix_match_unique'),
    )

class DbStore:
    def __init__(self, db_path: str = "sqlite:///football_engine.db"):
        self.engine = create_engine(db_path, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_all(self):
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        return self.SessionLocal()
