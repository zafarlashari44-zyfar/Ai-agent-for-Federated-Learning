-- Demo ECG remapping: 48 unique MIT-BIH records for ECG-0001..ECG-0048.
-- ECG-0049 and ECG-0050 intentionally retain duplicate demo mappings
-- because MIT-BIH Arrhythmia Database contains 48 records.
--
-- This migration assumes patient_ecg_records already exists.

BEGIN;

WITH mit_records AS (
    SELECT *
    FROM (
        VALUES
            (1,'100'), (2,'101'), (3,'102'), (4,'103'),
            (5,'104'), (6,'105'), (7,'106'), (8,'107'),
            (9,'108'), (10,'109'), (11,'111'), (12,'112'),
            (13,'113'), (14,'114'), (15,'115'), (16,'116'),
            (17,'117'), (18,'118'), (19,'119'), (20,'121'),
            (21,'122'), (22,'123'), (23,'124'), (24,'200'),
            (25,'201'), (26,'202'), (27,'203'), (28,'205'),
            (29,'207'), (30,'208'), (31,'209'), (32,'210'),
            (33,'212'), (34,'213'), (35,'214'), (36,'215'),
            (37,'217'), (38,'219'), (39,'220'), (40,'221'),
            (41,'222'), (42,'223'), (43,'228'), (44,'230'),
            (45,'231'), (46,'232'), (47,'233'), (48,'234')
    ) AS r(seq, record_id)
),
patient_rows AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY demo_ecg_id) AS seq
    FROM patient_ecg_records
    WHERE is_active = true
      AND mapping_type = 'demo_research'
)
UPDATE patient_ecg_records e
SET
    dataset      = 'MIT-BIH Arrhythmia Database',
    record_id    = r.record_id,
    hea_filename = r.record_id || '.hea',
    dat_filename = r.record_id || '.dat'
FROM patient_rows p
JOIN mit_records r
    ON r.seq = p.seq
WHERE e.id = p.id;

COMMIT;
