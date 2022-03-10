PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL, 
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
CREATE TABLE pieces (
	id INTEGER NOT NULL, 
	title VARCHAR, 
	path VARCHAR, 
	PRIMARY KEY (id)
);
INSERT INTO pieces VALUES(1,'Haydn_Sonata_Hob._XVI37_Mov._1_D_Major','./resources/midi/full/Haydn_Sonata_Hob._XVI37_Mov._1_D_Major.mid');
INSERT INTO pieces VALUES(2,'cmaj','./resources/midi/full/cmaj.mid');
INSERT INTO pieces VALUES(3,'Happy_Birthday_To_You_C_Major','./resources/midi/full/Happy_Birthday_To_You_C_Major.mid');
CREATE TABLE subpieces (
	id INTEGER NOT NULL, 
	title VARCHAR, 
	path VARCHAR, 
	piece_id INTEGER, 
	PRIMARY KEY (id), 
	FOREIGN KEY(piece_id) REFERENCES pieces (id)
);
INSERT INTO subpieces VALUES(1,'Happy_Birthday_To_You_C_Major-Part2','./resources/midi/subpieces/Happy_Birthday_To_You_C_Major-Part2.mid',3);
CREATE TABLE schedules (
	id INTEGER NOT NULL, 
	start_measure INTEGER, 
	end_measure INTEGER, 
	player VARCHAR, 
	piece_id INTEGER, 
	subpiece_id INTEGER, 
	PRIMARY KEY (id), 
	FOREIGN KEY(piece_id) REFERENCES pieces (id), 
	FOREIGN KEY(subpiece_id) REFERENCES subpieces (id)
);
INSERT INTO schedules VALUES(1,0,9,'Pianist',3,NULL);
INSERT INTO schedules VALUES(2,10,18,'VirtuosoNet',3,1);
CREATE INDEX ix_pieces_title ON pieces (title);
CREATE INDEX ix_pieces_id ON pieces (id);
CREATE INDEX ix_subpieces_id ON subpieces (id);
CREATE INDEX ix_schedules_id ON schedules (id);
CREATE INDEX ix_schedules_end_measure ON schedules (end_measure);
CREATE INDEX ix_schedules_start_measure ON schedules (start_measure);
COMMIT;
