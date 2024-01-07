import sqlite3


class SQLiteProvider(object):

    @staticmethod
    def write(data_it, target, data2col_func, table_name, columns, sqlite3_column_types,
              id_column_name="id", id_column_type="TEXT"):
        assert(callable(data2col_func))
        assert(isinstance(columns, list))
        assert(isinstance(sqlite3_column_types, list))
        assert(len(columns) == len(sqlite3_column_types))

        with sqlite3.connect(target) as con:
            cur = con.cursor()

            # Provide the ID column.
            columns = [id_column_name] + columns
            sqlite3_column_types = [id_column_type] + sqlite3_column_types

            # Compose the whole columns list.
            content = ", ".join([" ".join(item) for item in zip(columns, sqlite3_column_types)])
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name}({content})")
            cur.execute(f"CREATE INDEX IF NOT EXISTS i_id ON {table_name}({id_column_name})")

            for uid, data in data_it:
                r = cur.execute(f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE {id_column_name}='{uid}');")
                ans = r.fetchone()[0]
                if ans == 1:
                    continue

                params = ", ".join(tuple(['?'] * (len(columns))))
                cur.execute(f"INSERT INTO {table_name} VALUES ({params})", [str(uid)] + data2col_func(data))
                con.commit()

            cur.close()
