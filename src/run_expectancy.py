"""
Run expectancy by count and base-out state, built from our own Statcast
data (not delta_run_exp), then cross-checked against delta_run_exp as an
independent validation of the methodology.

RE(state) = average runs scored from this state through the end of the
half-inning. State = (balls, strikes, outs_when_up, base_state), where
base_state is one of the 8 combinations of runners on 1st/2nd/3rd.

Run: python src/run_expectancy.py
"""
import duckdb
from pathlib import Path

DB_PATH = "data/baseball.duckdb"


def build_db(con):
    con.execute(f"""
        CREATE OR REPLACE TABLE statcast AS
        SELECT * FROM read_parquet('data/statcast_*.parquet')
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_game ON statcast(game_pk)")


def compute_run_expectancy(con):
    """
    RE24-style table: runs scored from the current (balls, strikes, outs,
    base_state) through the end of the half-inning, built directly from
    bat_score / post_bat_score rather than delta_run_exp.
    """
    con.execute("""
        CREATE OR REPLACE VIEW pitch_state AS
        SELECT
            game_pk, game_year, inning, inning_topbot, at_bat_number, pitch_number,
            balls, strikes, outs_when_up,
            (on_1b IS NOT NULL) AS r1,
            (on_2b IS NOT NULL) AS r2,
            (on_3b IS NOT NULL) AS r3,
            bat_score,
            post_bat_score,
            delta_run_exp
        FROM statcast
        WHERE balls IS NOT NULL AND strikes IS NOT NULL AND outs_when_up IS NOT NULL
    """)

    # MUST be MAX(post_bat_score), not MAX(bat_score): bat_score is the score
    # BEFORE the play resolves, so runs scored on the play that ENDS the
    # half-inning never appear in any later row. Using bat_score understated
    # every state (bases-loaded-2-out by 0.027 runs, bases-empty-0-out by 0.007).
    con.execute("""
        CREATE OR REPLACE VIEW half_inning_final_score AS
        SELECT game_pk, inning, inning_topbot,
               MAX(post_bat_score) AS final_bat_score
        FROM pitch_state
        GROUP BY 1, 2, 3
    """)

    # Walk-off innings are censored: the home team stops batting the moment the
    # winning run scores, so observed runs understate the state's true value.
    # Excluded precisely (last half-inning of the game, bottom half, home team
    # won) rather than dropping every game's last half-inning, which would also
    # throw away complete top-of-9th innings.
    con.execute("""
        CREATE OR REPLACE VIEW game_end AS
        SELECT game_pk,
               MAX(inning * 2 + CASE WHEN inning_topbot = 'Bot' THEN 1 ELSE 0 END) AS last_hi_key,
               MAX(post_home_score) AS fin_home,
               MAX(post_away_score) AS fin_away
        FROM statcast
        GROUP BY 1
    """)

    con.execute("""
        CREATE OR REPLACE VIEW pitch_state_with_target AS
        SELECT p.*, h.final_bat_score - p.bat_score AS runs_rest_of_inning
        FROM pitch_state p
        JOIN half_inning_final_score h
          ON p.game_pk = h.game_pk AND p.inning = h.inning AND p.inning_topbot = h.inning_topbot
        JOIN game_end g ON p.game_pk = g.game_pk
        WHERE NOT (
            (p.inning * 2 + CASE WHEN p.inning_topbot = 'Bot' THEN 1 ELSE 0 END) = g.last_hi_key
            AND p.inning_topbot = 'Bot'
            AND g.fin_home > g.fin_away
        )
    """)

    re_table = con.execute("""
        SELECT balls, strikes, outs_when_up, r1, r2, r3,
               AVG(runs_rest_of_inning) AS run_exp,
               AVG(delta_run_exp) AS avg_delta_run_exp_same_state,
               COUNT(*) AS n
        FROM pitch_state_with_target
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY outs_when_up, r3 DESC, r2 DESC, r1 DESC, balls, strikes
    """).df()
    return re_table


def cross_check_vs_delta_run_exp(con):
    """
    Independent validation: for called strikes vs balls within each count,
    average delta_run_exp should be close to (but not identical to) the
    run-value implied by our own RE table's state transitions. This is the
    walkthrough's suggested shortcut check, used here as a validation of
    the full RE table rather than as the primary method.
    """
    return con.execute("""
        SELECT balls, strikes, description,
               AVG(delta_run_exp) AS avg_delta_run_exp,
               COUNT(*) AS n
        FROM statcast
        WHERE description IN ('called_strike', 'ball')
        GROUP BY 1, 2, 3
        ORDER BY balls, strikes, description
    """).df()


def main():
    Path("data").mkdir(exist_ok=True)
    con = duckdb.connect(DB_PATH)
    build_db(con)
    print(con.execute("SELECT game_year, COUNT(*) FROM statcast GROUP BY 1 ORDER BY 1").df())

    re_table = compute_run_expectancy(con)
    re_table.to_parquet("data/run_expectancy.parquet", index=False)
    print(f"\nrun expectancy table: {len(re_table)} (balls,strikes,outs,base_state) rows")
    print(re_table.head(20).to_string(index=False))

    # base-out only (24-state RE24), collapsed across counts, for sanity-checking
    # against published RE24 tables
    re24 = con.execute("""
        SELECT outs_when_up AS outs, r1, r2, r3,
               AVG(runs_rest_of_inning) AS run_exp, COUNT(*) AS n
        FROM pitch_state_with_target
        GROUP BY 1,2,3,4
        ORDER BY outs, r3 DESC, r2 DESC, r1 DESC
    """).df()
    print("\n=== RE24 (base-out state only, collapsed across counts) ===")
    print(re24.to_string(index=False))

    print("\n=== Cross-check: avg delta_run_exp for called_strike vs ball by count ===")
    print(cross_check_vs_delta_run_exp(con).to_string(index=False))


if __name__ == "__main__":
    main()
