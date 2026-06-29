CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    output_filename TEXT NOT NULL,
    objects_count INTEGER DEFAULT 0,
    objects_list TEXT DEFAULT '[]',
    average_confidence REAL DEFAULT 0.0,
    inference_time REAL DEFAULT 0.0,
    model_name TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
