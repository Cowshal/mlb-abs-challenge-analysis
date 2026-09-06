"""
Build the "real games from 2026" case studies for the app.

Everything on the site is aggregate. This turns the same per-pitch scoring the
model already does -- P(the original call was wrong) and dre (runs at stake if
the call flips) -- into named, dated at-bats a reader can check.

Three categories, all drawn from data/challenge_opportunities.parquet (every
called ball/strike in 2026, each scored with p_success = P(call was wrong) and
dre) plus data/abs_challenges.parquet (what teams actually did):

  1. missed        -- high P(wrong), high dre, the challenging side still had a
                      challenge in hand, and nobody challenged. Ranked by
                      expected runs forfeited, net of the option-value cost of
                      spending a challenge token here (ev_net).
  2. burn          -- one game where a side spent BOTH challenges on
                      low-leverage calls, wrongly, then hit a high-value call
                      later with nothing left. Ranked by the leverage gap
                      between what they couldn't challenge and what they burned.
  3. endorsed_win  -- a challenge that was genuinely a coin flip (P(wrong) well
                      under certainty) but high enough leverage that the optimal
                      policy still says fire -- and it won. The model isn't only
                      criticizing.

For the top few in each category we pull the rest of that plate appearance and
half-inning from the Stats API feed/live endpoint, purely as after-the-fact
colour. A missed challenge is a mistake because of the decision at the time
(expected runs forfeited), NOT because of what happened next -- the app text
says this explicitly and the outcome is labelled as such here.

Challenge-token accounting matches src/abs_policy.py exactly: two challenges to
start, only an INCORRECT one is spent, rights gone after two incorrect, one
restored at the start of each extra inning (k -> max(k - (inning-9), 0)).

Run: python scripts/build_case_studies.py
Output: data/case_studies.parquet  (copied to app/data/ by build_app_data.py)
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries
from geometry import center_distance_to_zone, ball_edge_distance

SEASON = 2026
CACHE = Path("data/_feedlive_cache")
OPP_PATH = "data/challenge_opportunities.parquet"
CH_PATH = "data/abs_challenges.parquet"
OV_PATH = "data/option_values.parquet"

# How many cases per category to publish. The app renders the first few in full
# (with a zone plot and a "what happened next" paragraph) and lists the rest.
N_MISSED = 6
N_ENDORSED = 6

# Category-3 "far from certain" band: a challenge whose modelled P(wrong) sat
# here was a real judgement call, not a gimme -- these are the ones that show
# the optimal policy firing without being sure.
ENDORSED_LO, ENDORSED_HI = 0.45, 0.78


# ----------------------------------------------------------------------------- #
#  side / team helpers                                                          #
# ----------------------------------------------------------------------------- #
def opp_side(challenger, topbot):
    """home/away of the side that could challenge -- matches build_app_data.team_of."""
    top = np.asarray(topbot) == "Top"
    batting = np.asarray(challenger) == "batting"
    # batting team bats in Bot halves; fielding team fields in Bot halves
    return np.where((batting & ~top) | (~batting & top), "home", "away")


def challenge_side(player_type, half_inning):
    """home/away of the team that made an actual challenge."""
    top = half_inning == "top"
    batting = player_type == "batter"
    # batting team challenges called strikes; fielding team called balls
    if batting:
        return "away" if top else "home"
    return "home" if top else "away"


def base_out_label(r1, r2, r3, outs):
    bases = [b for b, on in ((("1st"), r1), ("2nd", r2), ("3rd", r3)) if on]
    if not bases:
        base = "bases empty"
    elif len(bases) == 3:
        base = "bases loaded"
    else:
        base = " & ".join(bases)
    return f"{base}, {outs} out" + ("s" if outs != 1 else "")


# ----------------------------------------------------------------------------- #
#  Stats API: what happened next                                               #
# ----------------------------------------------------------------------------- #
def feed_live(game_pk):
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{game_pk}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    r = get_with_retries(
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
    fp.write_text(r.text)
    return r.json()


def whats_next(game_pk, at_bat_number, inning, half_l, batting_team, kind):
    """Rest of the PA and half-inning from the live feed, plus the final line.
    Returns a plain-language paragraph. Purely descriptive -- carries no claim
    that the outcome validates or refutes the decision. `half_l` is 'top' or
    'bottom'; `batting_team` is whichever side was hitting during this PA."""
    try:
        feed = feed_live(game_pk)
        plays = feed["liveData"]["plays"]["allPlays"]
        idx = at_bat_number - 1
        pos = next((i for i, p in enumerate(plays)
                    if p["about"]["atBatIndex"] == idx), None)
        if pos is None:
            return ""
        pa = plays[pos]
        res = pa["result"]
        ev, desc = res.get("event", ""), res.get("description", "") or ""

        def bat_score(p):
            return (p["result"]["awayScore"] if half_l == "top"
                    else p["result"]["homeScore"])

        score_before = bat_score(plays[pos - 1]) if pos > 0 else 0
        same = [p for p in plays[pos:]
                if p["about"]["inning"] == inning
                and p["about"]["halfInning"] == half_l]
        runs_rest = bat_score(same[-1]) - score_before if same else 0

        ls = feed["liveData"]["linescore"]["teams"]
        fa, fh = ls["away"].get("runs", 0), ls["home"].get("runs", 0)
        home_abbr = feed["gameData"]["teams"]["home"]["abbreviation"]
        away_abbr = feed["gameData"]["teams"]["away"]["abbreviation"]
        winner = home_abbr if fh > fa else away_abbr if fa > fh else "tie"
        margin = abs(fh - fa)

        pa_txt = (desc.strip() if desc else ev).replace(" (pitch result)", "")
        if kind == "endorsed_win":
            # strip "X challenged, call on the field was overturned:" -- the lead
            # already says the challenge won.
            import re
            pa_txt = re.sub(r"^.*?overturned:\s*", "", pa_txt)
            lead = "The challenge won; the corrected call: "
        else:
            lead = "No challenge was made. The plate appearance ended: "
        inning_txt = (
            f"{batting_team} did not score again in the inning after this point."
            if runs_rest == 0 else
            f"{batting_team} scored {runs_rest} more run"
            f"{'s' if runs_rest != 1 else ''} in the inning after this point.")
        final_txt = (f"Final: {away_abbr} {fa}, {home_abbr} {fh}"
                     + (f" ({winner} by {margin})." if winner != "tie" else "."))
        return f"{lead}{pa_txt} {inning_txt} {final_txt}"
    except Exception as e:  # noqa: BLE001 -- colour text, never load-bearing
        print(f"  whats_next({game_pk}) failed: {type(e).__name__}: {e}")
        return ""


# ----------------------------------------------------------------------------- #
#  names                                                                       #
# ----------------------------------------------------------------------------- #
def fetch_names(ids):
    names = {}
    ids = [int(i) for i in sorted(set(int(x) for x in ids))]
    for i in range(0, len(ids), 50):
        r = get_with_retries(
            "https://statsapi.mlb.com/api/v1/people",
            params={"personIds": ",".join(str(b) for b in ids[i:i + 50])})
        for p in r.json().get("people", []):
            names[p["id"]] = p.get("fullName")
    return names


# ----------------------------------------------------------------------------- #
#  follow-up analyses: distributions behind the tab's plain-language claims     #
# ----------------------------------------------------------------------------- #
def analysis_extras(opp, abbr, model_version, generated_at):
    """Back the three claims in the tab with distributions, not anecdotes:
      1. the 3-2 concentration among model-endorsed missed opportunities;
      2. how far those ~16k misses actually clear the challenge threshold
         (hairline vs. comfortable), so the count isn't read as overclaiming;
      3. whether the genuinely-uncertain, high-leverage *winning* challenges
         cluster by team and by role (a link to the catcher finding, or not).
    """
    from scipy.stats import pearsonr, spearmanr

    n_games = opp.game_pk.nunique()
    opp = opp.copy()
    opp["team_abbr"] = [abbr[int(g)][0 if s == "home" else 1]
                        for g, s in zip(opp.game_pk, opp.side)]
    games_long = pd.DataFrame(
        [(g, h) for g, (h, a) in abbr.items()]
        + [(g, a) for g, (h, a) in abbr.items()], columns=["game_pk", "team_abbr"])
    games_per = games_long.groupby("team_abbr").game_pk.nunique().rename("games")

    def stamp(d):
        d = d.copy()
        d["model_version"] = model_version
        d["generated_at"] = generated_at
        return d

    # ---- 1 + 2: model-endorsed missed opportunities ----
    miss = opp[(~opp.was_challenged) & (opp.k_used < 2) & opp.endorsed].copy()
    n = len(miss)
    miss["count_label"] = miss.balls.astype(str) + "-" + miss.strikes.astype(str)
    miss["full_count"] = (miss.balls == 3) & (miss.strikes == 2)
    ev = miss.ev_net.values

    LARGE = 0.50   # "large" missed opportunity: >= half a run forfeited
    TOP_N = 100

    def count_table(frame):
        t = (frame.groupby("count_label")
             .agg(n=("ev_net", "size"), mean_ev_net=("ev_net", "mean"),
                  mean_dre=("dre", "mean"), mean_p_wrong=("p_wrong", "mean"),
                  mean_p_star=("p_star", "mean"))
             .reset_index())
        t["pct"] = t.n / len(frame)
        t["is_full_count"] = t.count_label == "3-2"
        return t.sort_values("n", ascending=False)

    # (a) all endorsed misses -- 0-0 dominates by sheer volume; this is the
    #     honest baseline. (b) the LARGE tail and (c) the TOP_N by forfeited
    #     runs -- this is where the 3-2 concentration actually lives.
    by_count_all = count_table(miss).assign(scope="all")
    by_count_large = count_table(miss[miss.ev_net >= LARGE]).assign(scope="large")
    by_count_top = count_table(
        miss.sort_values("ev_net", ascending=False).head(TOP_N)).assign(scope="top100")
    by_count = pd.concat([by_count_all, by_count_large, by_count_top],
                         ignore_index=True)
    stamp(by_count).to_parquet("data/endorsed_miss_by_count.parquet", index=False)

    fc = miss[miss.full_count]
    summary = pd.DataFrame([{
        "n": n, "per_game": n / n_games,
        "ev_net_q1": float(np.percentile(ev, 25)),
        "ev_net_median": float(np.median(ev)),
        "ev_net_q3": float(np.percentile(ev, 75)),
        "ev_net_p90": float(np.percentile(ev, 90)),
        "ev_net_mean": float(ev.mean()),
        "frac_hair": float((ev < 0.05).mean()),
        "frac_modest": float(((ev >= 0.05) & (ev < 0.20)).mean()),
        "frac_comfortable": float(((ev >= 0.20) & (ev < LARGE)).mean()),
        "frac_large": float((ev >= LARGE).mean()),
        "n_comfortable_plus": int((ev >= 0.20).sum()),
        "n_comfortable_plus_per_game": float((ev >= 0.20).sum() / n_games),
        "n_large": int((ev >= LARGE).sum()),
        "games_per_large": float(n_games / max((ev >= LARGE).sum(), 1)),
        # a small ev_net is low STAKES, not a marginal decision: the median
        # miss still clears its break-even by this many points of confidence.
        "conf_margin_median": float(np.median((miss.p_wrong - miss.p_star).values)),
        # 3-2 concentration, by slice
        "full_count_share_all": float(miss.full_count.mean()),
        "full_count_share_large": float(miss[miss.ev_net >= LARGE].full_count.mean()),
        "full_count_share_top100": float(
            miss.sort_values("ev_net", ascending=False).head(TOP_N).full_count.mean()),
        "full_count_mean_p_star": float(fc.p_star.mean()),
        "full_count_mean_dre": float(fc.dre.mean()),
        "non_full_mean_dre": float(miss[~miss.full_count].dre.mean()),
    }])
    stamp(summary).to_parquet("data/endorsed_miss_summary.parquet", index=False)

    edges = np.array([0, .02, .04, .06, .08, .10, .15, .20, .30, .50, .75, 3.0])
    counts, _ = np.histogram(ev, bins=edges)
    hist = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "n": counts})
    stamp(hist).to_parquet("data/endorsed_miss_evnet_hist.parquet", index=False)

    print(f"\n=== model-endorsed missed opportunities (n={n:,}, "
          f"{n / n_games:.1f}/game) ===")
    for scope in ("all", "large", "top100"):
        s = by_count[by_count.scope == scope]
        print(f"  [{scope}] n={s.n.sum()}  "
              + "  ".join(f"{r.count_label}:{r.pct:.0%}" for r in s.head(4).itertuples()))
    print(summary.T.to_string())

    # ---- 3: do the coin-flip winning challenges cluster? ----
    band = opp.p_wrong.between(ENDORSED_LO, ENDORSED_HI)
    opp["cf_attempt"] = opp.was_challenged & band & opp.endorsed
    opp["cf_win"] = opp.cf_attempt & opp.overturned
    opp["cf_attempt_hi"] = opp.cf_attempt & (opp.dre >= 0.75)  # genuinely high-lev
    opp["cf_win_hi"] = opp.cf_attempt_hi & opp.overturned

    made = opp[opp.was_challenged]
    # runs gained through successful challenges, split by whether the call was
    # a coin flip. cf_attempts correlating with cf-band runs is partly
    # mechanical (a won cf challenge IS runs); correlating it with runs gained
    # on the OTHER challenges is the de-circularised "this team just challenges
    # well" test.
    won = made[made.overturned]
    won_cf = won[won.p_wrong.between(ENDORSED_LO, ENDORSED_HI)]
    runs_cf = won_cf.groupby("team_abbr").dre.sum().rename("runs_gained_cf")
    runs_other = (won[~won.p_wrong.between(ENDORSED_LO, ENDORSED_HI)]
                  .groupby("team_abbr").dre.sum().rename("runs_gained_other"))
    runs_all = won.groupby("team_abbr").dre.sum().rename("runs_gained")

    cf = (opp.groupby("team_abbr")
          .agg(cf_attempts=("cf_attempt", "sum"), cf_wins=("cf_win", "sum"),
               cf_attempts_hi=("cf_attempt_hi", "sum"),
               cf_wins_hi=("cf_win_hi", "sum"),
               total_challenges=("was_challenged", "sum"))
          .reset_index()
          .merge(games_per, on="team_abbr"))
    for s in (runs_all, runs_cf, runs_other):
        cf = cf.merge(s, on="team_abbr", how="left")
    cf[["runs_gained", "runs_gained_cf", "runs_gained_other"]] = \
        cf[["runs_gained", "runs_gained_cf", "runs_gained_other"]].fillna(0.0)
    cf["cf_attempts_per_game"] = cf.cf_attempts / cf.games
    cf["cf_win_rate"] = cf.cf_wins / cf.cf_attempts.replace(0, np.nan)
    cf["cf_share_of_challenges"] = cf.cf_attempts / cf.total_challenges
    cf["runs_gained_per_game"] = cf.runs_gained / cf.games
    cf["runs_other_per_game"] = cf.runs_gained_other / cf.games
    cf["runs_rank"] = cf.runs_gained.rank(ascending=False).astype(int)
    cf = cf.sort_values("runs_gained", ascending=False)
    stamp(cf).to_parquet("data/coinflip_by_team.parquet", index=False)

    sel = opp[opp.was_challenged & band & opp.endorsed]
    role = (sel.groupby("challenger")
            .agg(attempts=("overturned", "size"), wins=("overturned", "sum"))
            .reset_index())
    role["win_rate"] = role.wins / role.attempts
    tot_role = made.groupby("challenger").size().rename("role_total_challenges")
    role = role.merge(tot_role, on="challenger")
    role["share_of_role_challenges"] = role.attempts / role.role_total_challenges
    role["role_overall_win_rate"] = (
        made.groupby("challenger").overturned.mean().reindex(role.challenger).values)
    stamp(role).to_parquet("data/coinflip_by_role.parquet", index=False)

    def rp(a, b, method=pearsonr):
        m = np.isfinite(a) & np.isfinite(b)
        r, p = method(np.asarray(a)[m], np.asarray(b)[m])
        return float(r), float(p)

    r_att, p_att = rp(cf.cf_attempts_per_game, cf.runs_gained_per_game)
    r_dec, p_dec = rp(cf.cf_attempts_per_game, cf.runs_other_per_game)  # de-circ.
    r_rate, p_rate = rp(cf.cf_win_rate, cf.runs_other_per_game)
    r_hi, p_hi = rp(cf.cf_attempts_hi / cf.games, cf.runs_gained_per_game)
    r_sh, p_sh = rp(cf.cf_share_of_challenges, cf.runs_gained, spearmanr)
    top5_runs = set(cf.head(5).team_abbr)
    top5_cf = set(cf.sort_values("cf_attempts_per_game", ascending=False).head(5).team_abbr)
    cf_summary = pd.DataFrame([{
        "n_teams": len(cf),
        "cf_attempts_total": int(cf.cf_attempts.sum()),
        "cf_wins_total": int(cf.cf_wins.sum()),
        "cf_win_rate_overall": float(cf.cf_wins.sum() / cf.cf_attempts.sum()),
        "cf_attempts_hi_total": int(cf.cf_attempts_hi.sum()),
        "r_attempts_runs": r_att, "p_attempts_runs": p_att,
        "r_attempts_runs_other": r_dec, "p_attempts_runs_other": p_dec,
        "r_winrate_runs_other": r_rate, "p_winrate_runs_other": p_rate,
        "r_attempts_hi_runs": r_hi, "p_attempts_hi_runs": p_hi,
        "r_share_runs_spearman": r_sh, "p_share_runs_spearman": p_sh,
        "cf_attempts_per_game_mean": float(cf.cf_attempts_per_game.mean()),
        "cf_attempts_per_game_max": float(cf.cf_attempts_per_game.max()),
        "top5_runs_teams": ", ".join(cf.head(5).team_abbr),
        "top5_by_cf_rate_teams": ", ".join(
            cf.sort_values("cf_attempts_per_game", ascending=False).head(5).team_abbr),
        "top5_overlap": len(top5_runs & top5_cf),
        "role_fielding_share": float(
            role.loc[role.challenger == "fielding", "share_of_role_challenges"].iloc[0]),
        "role_batting_share": float(
            role.loc[role.challenger == "batting", "share_of_role_challenges"].iloc[0]),
    }])
    stamp(cf_summary).to_parquet("data/coinflip_summary.parquet", index=False)

    print(f"\n=== coin-flip endorsed challenges (p_wrong in "
          f"[{ENDORSED_LO}, {ENDORSED_HI}]) ===")
    print(role.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(cf_summary.T.to_string())
    print("\nby team (top 10 by runs gained):")
    print(cf[["team_abbr", "runs_rank", "games", "cf_attempts", "cf_wins",
              "cf_win_rate", "cf_attempts_per_game", "runs_gained",
              "runs_gained_other"]].head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ----------------------------------------------------------------------------- #
#  main                                                                        #
# ----------------------------------------------------------------------------- #
def main():
    con = duckdb.connect("data/baseball.duckdb")
    games = con.execute(f"""
        SELECT DISTINCT game_pk, home_team, away_team
        FROM statcast WHERE game_year = {SEASON}
    """).df()
    games["game_pk"] = games.game_pk.astype(np.int64)
    abbr = {int(r.game_pk): (r.home_team, r.away_team) for r in games.itertuples()}

    opp = pd.read_parquet(OPP_PATH)
    opp["game_pk"] = opp.game_pk.astype(np.int64)
    opp["side"] = opp_side(opp.challenger.values, opp.inning_topbot.values)
    opp["t"] = (opp.inning - 1) * 2 + np.where(opp.inning_topbot == "Top", 1, 2)
    opp["t"] = opp.t.clip(upper=30)
    opp["pos"] = list(zip(opp.inning, opp.at_bat_number, opp.pitch_number))

    # signed ball-edge distance (ft); negative for batting = strike call was
    # wrong, positive for fielding = ball call was wrong. p_success already
    # encodes the correct direction, this is only for the "missed by X in" line.
    opp["d_ft"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    opp["p_wrong"] = opp.p_success  # model P(original call was wrong)

    # ---- option value C(k) and break-even p* per opportunity ----
    ov = pd.read_parquet(OV_PATH).set_index("t")[["C_k0", "C_k1"]]
    last_t = ov.index.max()

    # ---- reconstruct incorrect-challenge timeline per (game, side) ----
    ch = pd.read_parquet(CH_PATH)
    ch["game_pk"] = ch.game_pk.astype(np.int64)
    ch["side"] = [challenge_side(pt, hi)
                  for pt, hi in zip(ch.challenging_player_type, ch.half_inning)]
    ch["pos"] = list(zip(ch.inning, ch.ab_number, ch.pitch_number))
    incorrect = ch[~ch.is_overturned]
    inc_pos = {(g, s): sorted(sub.pos.tolist())
               for (g, s), sub in incorrect.groupby(["game_pk", "side"])}

    def k_used_before(game_pk, side, pos, inning):
        """Incorrect challenges the side had spent *before* this pitch, after
        crediting one back per extra inning already reached."""
        positions = inc_pos.get((int(game_pk), side), [])
        raw = sum(1 for p in positions if p < pos)
        restored = max(0, inning - 9)
        return max(0, raw - restored)

    k_arr = np.array([k_used_before(g, s, p, i) for g, s, p, i in
                      zip(opp.game_pk, opp.side, opp.pos, opp.inning)])
    opp["k_used"] = k_arr
    opp["challenges_remaining"] = (2 - k_arr).clip(0, 2)

    # Evaluate the challenge on its merits at the token cost the side faces:
    # k=0 -> C_k0, k>=1 -> C_k1 (the last-token cost). For an already-exhausted
    # side (k>=2) this answers "would the optimal policy have wanted this
    # challenge, had one been available" -- the quantity category 2 is about.
    k_eval = np.minimum(k_arr, 1)
    C = np.where(k_eval == 0,
                 ov.reindex(opp.t).C_k0.fillna(ov.loc[last_t].C_k0).values,
                 ov.reindex(opp.t).C_k1.fillna(ov.loc[last_t].C_k1).values)
    opp["opt_cost"] = C
    opp["p_star"] = C / (opp.dre.values + C)
    opp["ev_gross"] = opp.p_wrong.values * opp.dre.values
    opp["ev_net"] = opp.p_wrong.values * opp.dre.values - (1 - opp.p_wrong.values) * C
    opp["endorsed"] = opp.ev_net > 0

    # =====================================================================
    #  Category 1: biggest missed opportunities
    # =====================================================================
    missable = opp[(~opp.was_challenged) & (opp.k_used < 2) & opp.endorsed].copy()
    n_missed_total = len(missable)
    n_missed_games = missable.game_pk.nunique()
    # one per game for variety; keep the worst miss in each
    missable = missable.sort_values("ev_net", ascending=False)
    top_missed = missable.drop_duplicates("game_pk").head(N_MISSED).copy()
    top_missed["category"] = "missed"
    top_missed["rank"] = range(1, len(top_missed) + 1)
    tail_lo = missable.ev_net.iloc[min(99, len(missable) - 1)]
    top_missed["comparable_n"] = n_missed_total
    top_missed["comparable_desc"] = (
        f"{n_missed_total:,} called pitches in 2026 where the model says "
        f"challenge and the side still had a challenge in hand but didn't use "
        f"it (across {n_missed_games:,} games). The vast majority are "
        f"low-stakes; the 100 largest each forfeit at least "
        f"{tail_lo:.2f} expected runs.")

    # =====================================================================
    #  Category 3: model-endorsed successful challenges that were coin flips
    # =====================================================================
    won = opp[opp.was_challenged & opp.overturned].copy()
    n_won = len(won)
    band = won[(won.p_wrong >= ENDORSED_LO) & (won.p_wrong <= ENDORSED_HI)
               & won.endorsed].copy()
    n_band = len(band)
    band = band.sort_values("dre", ascending=False)
    top_endorsed = band.drop_duplicates("game_pk").head(N_ENDORSED).copy()
    top_endorsed["category"] = "endorsed_win"
    top_endorsed["rank"] = range(1, len(top_endorsed) + 1)
    top_endorsed["comparable_n"] = n_band
    top_endorsed["comparable_desc"] = (
        f"{n_band:,} of the {n_won:,} challenges that were won in 2026 were "
        f"genuine judgement calls by this model's read "
        f"(P(wrong) between {ENDORSED_LO:.0%} and {ENDORSED_HI:.0%}) that the "
        f"optimal policy still endorsed. Ranked by runs at stake.")

    # =====================================================================
    #  Category 2: costly early burns -- spent both challenges, wrongly, on
    #  low-leverage calls, then hit a high-value call later with nothing left.
    # =====================================================================
    BURN_MAX_MEAN_EV = 0.15   # the burned challenges were near-nothing calls
    BURN_MAX_DRE = 0.75       # ...and none was a high-leverage situation
    LATER_MIN_PWRONG = 0.80   # the call they couldn't touch was clearly wrong
    LATER_MIN_EVNET = 0.50    # ...and genuinely high-value

    burn_rows = []
    for (game_pk, side), g in opp.groupby(["game_pk", "side"], sort=False):
        g = g.sort_values("pos")
        lost = g[g.was_challenged & ~g.overturned]
        if len(lost) < 2:
            continue  # never actually exhausted their rights this game/side
        second_burn_pos = sorted(lost.pos.tolist())[1]
        # the two challenges that actually cost them their rights
        burned = lost[lost.pos <= second_burn_pos]
        if burned.ev_gross.mean() >= BURN_MAX_MEAN_EV or burned.dre.max() >= BURN_MAX_DRE:
            continue  # they didn't waste them on low-leverage calls
        after = g[(g.pos > second_burn_pos) & (~g.was_challenged)
                  & (g.k_used >= 2) & (g.inning <= 9)      # regulation, no restore
                  & (g.p_wrong >= LATER_MIN_PWRONG) & (g.ev_net >= LATER_MIN_EVNET)]
        if after.empty:
            continue
        later = after.sort_values("ev_net", ascending=False).iloc[0]
        burn_rows.append((later.ev_net - burned.ev_gross.mean(),
                          game_pk, side, burned, later))

    burn_rows.sort(key=lambda r: r[0], reverse=True)
    n_burn_games = len(burn_rows)

    cases = pd.concat([top_missed, top_endorsed], ignore_index=True)

    burn_cases = []
    for rank, (gap, game_pk, side, burned, later) in enumerate(burn_rows[:3], 1):
        later = later.copy()
        later["category"] = "burn"
        later["rank"] = rank
        later["comparable_n"] = n_burn_games
        later["comparable_desc"] = (
            f"{n_burn_games} team-games in 2026 fit this pattern: both "
            f"challenges spent on calls worth under {BURN_MAX_MEAN_EV:.2f} "
            f"expected runs and lost, then a call at least {LATER_MIN_PWRONG:.0%} "
            f"likely wrong and worth {LATER_MIN_EVNET:.1f}+ runs later in "
            f"regulation with nothing left. Ranked by the leverage gap; this is "
            f"#{rank}.")
        later["leverage_gap"] = gap
        bd = []
        for b in burned.sort_values("pos").itertuples():
            half = "Top" if b.inning_topbot == "Top" else "Bot"
            bd.append(
                f"- **{b.inning}{half[0].lower()}**, {b.balls}-{b.strikes} count, "
                f"{base_out_label(b.r1, b.r2, b.r3, b.outs)} — challenged "
                f"({'batter' if b.challenger == 'batting' else 'catcher/pitcher'}) "
                f"and **lost**. Model read: {b.p_wrong:.0%} likely wrong, "
                f"{b.dre:.2f} runs at stake.")
        later["burn_detail"] = "\n".join(bd)
        burn_cases.append(later)
    if burn_cases:
        cases = pd.concat([cases, pd.DataFrame(burn_cases)], ignore_index=True)

    # =====================================================================
    #  enrich: names, teams, labels, and "what happened next"
    # =====================================================================
    ch_names = pd.concat([
        ch[["batter", "batter_name"]].rename(columns={"batter": "id", "batter_name": "nm"}),
        ch[["pitcher", "pitcher_name"]].rename(columns={"pitcher": "id", "pitcher_name": "nm"}),
    ]).drop_duplicates("id")
    name_map = dict(zip(ch_names.id.astype(int), ch_names.nm))
    missing = {int(x) for x in pd.concat([cases.batter, cases.pitcher])
               if int(x) not in name_map}
    if missing:
        name_map.update(fetch_names(missing))

    out = []
    for r in cases.itertuples():
        home_abbr, away_abbr = abbr[int(r.game_pk)]
        chal_team = home_abbr if r.side == "home" else away_abbr
        def_team = away_abbr if r.side == "home" else home_abbr
        role = r.challenger
        call_was = "strike" if r.original_call == "called_strike" else "ball"
        # signed miss in inches: how far past the boundary the pitch actually
        # was, on the side that makes the call wrong.
        miss_in = (-r.d_ft * 12) if role == "batting" else (r.d_ft * 12)
        half_l = "top" if r.inning_topbot == "Top" else "bottom"
        batting_team = chal_team if role == "batting" else def_team
        nxt = whats_next(int(r.game_pk), int(r.at_bat_number), int(r.inning),
                         half_l, batting_team, r.category)
        h = r.height_ft
        rec = {
            "category": r.category,
            "rank": int(r.rank),
            "game_pk": int(r.game_pk),
            "game_date": r.game_date,
            "away_team": away_abbr,
            "home_team": home_abbr,
            "challenging_side": r.side,
            "challenging_team": chal_team,
            "defending_team": def_team,
            "batting_team": batting_team,
            "inning": int(r.inning),
            "half": "Top" if r.inning_topbot == "Top" else "Bot",
            "balls": int(r.balls),
            "strikes": int(r.strikes),
            "outs": int(r.outs),
            "r1": bool(r.r1), "r2": bool(r.r2), "r3": bool(r.r3),
            "count_label": f"{r.balls}-{r.strikes}",
            "base_out_label": base_out_label(r.r1, r.r2, r.r3, r.outs),
            "batter_name": name_map.get(int(r.batter), str(r.batter)),
            "pitcher_name": name_map.get(int(r.pitcher), str(r.pitcher)),
            "role": role,
            "challenger_desc": "batter" if role == "batting" else "catcher / pitcher",
            "call_was": call_was,
            "challenges_remaining": int(r.challenges_remaining),
            "k_incorrect_used": int(r.k_used),
            "pitch_x_in": float(r.x_mid * 12),
            "pitch_z_in": float(r.z_mid * 12),
            "zone_top_in": float(0.535 * h * 12),
            "zone_bot_in": float(0.270 * h * 12),
            "batter_height_ft": float(h),
            "miss_in": float(miss_in),
            "p_wrong": float(r.p_wrong),
            "dre": float(r.dre),
            "ev_gross": float(r.ev_gross),
            "opt_cost": float(r.opt_cost),
            "ev_net": float(r.ev_net),
            "p_star": float(r.p_star),
            "endorsed": bool(r.endorsed),
            "comparable_n": int(r.comparable_n),
            "comparable_desc": r.comparable_desc,
            "leverage_gap": float(getattr(r, "leverage_gap", np.nan)),
            "burn_detail": (getattr(r, "burn_detail", "")
                            if isinstance(getattr(r, "burn_detail", ""), str) else ""),
            "narrative": nxt,
        }
        out.append(rec)

    df = pd.DataFrame(out)

    # inherit the policy model's provenance stamp (option_values is its output)
    ovstamp = pd.read_parquet(OV_PATH)
    df["model_version"] = ovstamp.model_version.iloc[0]
    df["generated_at"] = ovstamp.generated_at.iloc[0]

    df.to_parquet("data/case_studies.parquet", index=False)
    analysis_extras(opp, abbr, ovstamp.model_version.iloc[0],
                    ovstamp.generated_at.iloc[0])
    print(f"\nsaved {len(df)} cases -> data/case_studies.parquet")
    for cat in ("missed", "burn", "endorsed_win"):
        sub = df[df.category == cat]
        print(f"\n=== {cat} ({len(sub)}) ===")
        cols = ["rank", "game_date", "away_team", "home_team", "inning", "half",
                "count_label", "base_out_label", "batter_name", "p_wrong",
                "dre", "ev_net", "challenges_remaining"]
        print(sub[cols].to_string(index=False))
        if len(sub):
            print("  comparable:", sub.comparable_desc.iloc[0])


if __name__ == "__main__":
    main()
