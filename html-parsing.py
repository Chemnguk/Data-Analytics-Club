# %% [code]
# %% [code]
"""
Author: ABIODUN TIAMIYU
Date: 2025-07-09

Purpose:
This script extracts structured data from a large HTML file containing bet history records.
It parses and organizes key details such as Bet ID, Stake Amount, Odds, PnL (Profit and Loss),
Bet Type, Date, and number of Games per Bet.

Then it performs:
- Summary statistics for Stake and PnL
- Monthly aggregation
- Win ratio analytics
- Visualization (bar + pie charts)
- Clean export for client reporting
"""

from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
np.seterr(invalid='ignore')  

# Load the HTML content
with open('/kaggle/input/bet-history-of-user-1058/history_1058225375.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'lxml')

# 1. Extract Bet IDs
bet_ids_raw = []
for tag in soup.find_all('b', string=re.compile(r'Bet slip №\d+')):
    match = re.search(r'№(\d+)', tag.text)
    if match:
        bet_ids_raw.append(match.group(1))
unique_bet_ids = list(OrderedDict.fromkeys(bet_ids_raw))[:2509]
bet_ids = unique_bet_ids

# 2. Extract Dates
bet_dates = []
for div in soup.find_all('div', class_='cupHisNew')[:2509]:
    time_tag = div.find('time')
    bet_dates.append(time_tag.text.strip() if time_tag else None)

# 3. Extract Odds
bet_odds = []
for div in soup.find_all('div', class_='cupHisNew')[:2509]:
    cof_div = div.find('div', class_='hisCof')
    if cof_div:
        odds_text = cof_div.text.strip()
        try:
            bet_odds.append(float(odds_text))
        except ValueError:
            bet_odds.append(None)
    else:
        bet_odds.append(None)
print(f"Total odds found: {len(bet_odds)}")

# 4. Extract Stake Amounts
stake_amounts = []
for div in soup.find_all('div', class_='cupHisNew')[:2509]:
    try:
        table = div.find('table', class_='table_prop')
        if table:
            ngn_texts = table.find_all(string=re.compile(r'[\d,.]+\s*NGN'))
            if ngn_texts:
                match = re.search(r'([\d,.]+)\s*NGN', ngn_texts[0])
                if match:
                    stake = float(match.group(1).replace(',', ''))
                    stake_amounts.append(stake)
                else:
                    stake_amounts.append(None)
            else:
                stake_amounts.append(None)
        else:
            stake_amounts.append(None)
    except Exception:
        stake_amounts.append(None)

# 5. Extract PnL
pnl_values = []
for i, div in enumerate(soup.find_all('div', class_='cupHisNew')[:2509]):
    try:
        bold_tags = div.find_all('b')
        result_tag = bold_tags[-1]
        result_text = result_tag.text.strip()
        if result_text.lower() == "loss":
            pnl_values.append(-stake_amounts[i] if i < len(stake_amounts) and stake_amounts[i] is not None else None)
        else:
            match = re.search(r'([\d.,]+)', result_text)
            if match:
                amount = float(match.group(1).replace(',', ''))
                pnl_values.append(amount)
            else:
                pnl_values.append(None)
    except Exception:
        pnl_values.append(None)
# 6. Extract Bet Types
bet_types = []
for div in soup.find_all('div', class_='cupHisNew')[:2509]:
    try:
        table = div.find('table', class_='table_prop')
        if table:
            bet_type_cell = table.find('td', class_='ri')
            if bet_type_cell and "Bet type:" in bet_type_cell.text:
                match = re.search(r'Bet type:\s*(.+)', bet_type_cell.text)
                bet_types.append(match.group(1).strip() if match else None)
            else:
                bet_types.append(None)
        else:
            bet_types.append(None)
    except Exception:
        bet_types.append(None)

# 7. Extract Games per Bet
games_per_bet = []
status_keywords = ['win', 'loss', 'refund', 'void', 'cancelled', 'won', '%']
for div in soup.find_all('div', class_='cupHisNew')[:2509]:
    try:
        table = div.find('table', class_='table_prop')
        rows = table.find_all('tr') if table else []
        count = 0
        for row in rows[:-1]:
            tds = row.find_all('td')
            if tds:
                status_text = tds[-1].get_text(strip=True).lower()
                if any(keyword in status_text for keyword in status_keywords):
                    count += 1
        games_per_bet.append(count)
    except Exception:
        games_per_bet.append(0)

print(f"Total games_per_bet extracted: {len(games_per_bet)}")
print("Sample:", games_per_bet[:10])

# 8. Final DataFrame Build
lengths = list(map(len, [bet_ids, bet_dates, bet_odds, stake_amounts, pnl_values, bet_types, games_per_bet]))
if len(set(lengths)) == 1:
    df = pd.DataFrame({
        'Bet_Id': bet_ids,
        'Stake_Amount': stake_amounts,
        'Bet_Odds': bet_odds,
        'P_n_L': pnl_values,
        'Bet_Type': bet_types,
        'Date': pd.to_datetime(bet_dates, format='%d.%m.%Y | %H:%M', errors='coerce'),
        'Games_per_bet': games_per_bet
    })
    print(df.head())
else:
    print("Mismatch in column lengths:", lengths)

# ===========================
# Analytics and Visualization
# ===========================
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# --- Clean and Prepare ---
df = df.dropna(subset=['Stake_Amount', 'P_n_L'])

# Classify bet result
def classify_result(row):
    if row['P_n_L'] == row['Stake_Amount']:
        return 'refunded'
    elif row['P_n_L'] > row['Stake_Amount']:
        return 'win'
    elif row['P_n_L'] == 0 or row['P_n_L'] < row['Stake_Amount']:
        return 'lose'
    else:
        return 'unknown'

df['Result'] = df.apply(classify_result, axis=1)

# --- Core Metrics ---
total_bets = len(df)
wins = (df['Result'] == 'win').sum()
losses = (df['Result'] == 'lose').sum()
refunds = (df['Result'] == 'refunded').sum()

total_stake = df['Stake_Amount'].sum()
total_winnings = df['P_n_L'].sum()
net_profit = total_winnings - total_stake

win_ratio = (wins / total_bets) * 100
loss_ratio = (losses / total_bets) * 100
refund_ratio = (refunds / total_bets) * 100
profit_rate = (net_profit / total_stake) * 100 if total_stake else 0
roi = profit_rate

# --- Financial Behavior ---
avg_stake = df['Stake_Amount'].mean()
avg_win = df[df['Result'] == 'win']['P_n_L'].mean()
avg_loss = df[df['Result'] == 'lose']['Stake_Amount'].mean()
highest_stake = df['Stake_Amount'].max()
lowest_stake = df['Stake_Amount'].min()
largest_win = df['P_n_L'].max()
largest_loss = df[df['Result'] == 'lose']['Stake_Amount'].max()

# --- Date and Activity ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
total_days = df['Date'].dt.date.nunique()
bets_per_day = total_bets / total_days if total_days else 0

# --- Streaks ---
def longest_streak(series, target):
    streak = max_streak = 0
    for val in series:
        if val == target:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

longest_win_streak = longest_streak(df['Result'], 'win')
longest_loss_streak = longest_streak(df['Result'], 'lose')

# --- Summary Output ---
summary = {
    'Total Bets': total_bets,
    'Wins': wins,
    'Losses': losses,
    'Refunds': refunds,
    'Total Stake (NGN)': total_stake,
    'Total Winnings (NGN)': total_winnings,
    'Net Profit (NGN)': net_profit,
    'Win Ratio (%)': win_ratio,
    'Loss Ratio (%)': loss_ratio,
    'Refund Ratio (%)': refund_ratio,
    'Profit Rate (%)': profit_rate,
    'ROI (%)': roi,
    'Average Stake': avg_stake,
    'Average Win': avg_win,
    'Average Loss': avg_loss,
    'Highest Stake': highest_stake,
    'Lowest Stake': lowest_stake,
    'Largest Win': largest_win,
    'Largest Loss': largest_loss,
    'Total Betting Days': total_days,
    'Average Bets/Day': bets_per_day,
    'Longest Win Streak': longest_win_streak,
    'Longest Loss Streak': longest_loss_streak
}

# Create summary DataFrame
summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
summary_df.reset_index(inplace=True)
summary_df.columns = ['Metric', 'Value']

# === Create export folders ===
os.makedirs("exports/plots", exist_ok=True)
os.makedirs("exports/data", exist_ok=True)

# === Export summary metrics and data ===
summary_df.to_csv("exports/data/betting_metrics_summary.csv", index=False)
df.to_csv("exports/data/bet_history_data.csv", index=False)

# === 1. Pie Chart: Bet Result Distribution ===
result_counts = df['Result'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=140,
        colors=['green', 'red', 'gray'])
plt.title('Bet Result Distribution')
plt.axis('equal')
plt.tight_layout()
plt.savefig("exports/plots/result_distribution_pie.png")
plt.close()

# === 2. Bar Chart: Count of Result Types ===
plt.figure(figsize=(8, 5))
bars = plt.bar(result_counts.index, result_counts.values, color=['green', 'red', 'gray'])
plt.title('Number of Bet Outcomes')
plt.xlabel('Result Type')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, int(yval), ha='center', va='bottom')
plt.tight_layout()
plt.savefig("exports/plots/result_distribution_bar.png")
plt.close()

# === 3. Line Chart: Stake Over Time ===
stake_daily = df.groupby(df['Date'].dt.date)['Stake_Amount'].sum()
plt.figure(figsize=(10, 4))
plt.plot(stake_daily.index, stake_daily.values, marker='o', color='blue')
plt.title('Total Stake Over Time')
plt.xlabel('Date')
plt.ylabel('Stake (NGN)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("exports/plots/stake_over_time.png")
plt.close()

# === 4. Line Chart: Net Profit Per Day ===
df['Net_Profit'] = df['P_n_L'] - df['Stake_Amount']
profit_daily = df.groupby(df['Date'].dt.date)['Net_Profit'].sum()
plt.figure(figsize=(10, 4))
plt.plot(profit_daily.index, profit_daily.values, marker='o', color='purple')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Net Profit Per Day')
plt.xlabel('Date')
plt.ylabel('Net Profit (NGN)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("exports/plots/net_profit_per_day.png")
plt.close()

# === 5. Heatmap: Betting Activity by Hour and Day ===
df['Hour'] = df['Date'].dt.hour
df['Weekday'] = df['Date'].dt.day_name()
heatmap_data = df.groupby(['Weekday', 'Hour']).size().unstack(fill_value=0)

# Reorder weekdays
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(weekday_order)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5, annot=True, fmt='d')
plt.title('Betting Activity Heatmap (Hour vs Weekday)')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.tight_layout()
plt.savefig("exports/plots/betting_activity_heatmap.png")
plt.close()

print("✅ All exports completed successfully.")


# ==== Responsible Gambling Insight ====
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = df['Date'].dt.hour
df = df.sort_values('Date').reset_index(drop=True)

# ==== a. Daily Stake Threshold Breach (> NGN 10,000) ====
daily_stake = df.groupby(df['Date'].dt.date)['Stake_Amount'].sum()
risky_days = daily_stake[daily_stake > 10000]

# ==== b. Night-Time Betting Frequency (12AM–5AM) ====
night_bets = df[(df['Hour'] >= 0) & (df['Hour'] <= 5)]
night_bet_ratio = (len(night_bets) / len(df)) * 100

# ==== c. Chasing Losses Detection ====
df['Prev_Result'] = df['Result'].shift(1)
df['Prev_Stake'] = df['Stake_Amount'].shift(1)
df['Chasing_Loss'] = (df['Prev_Result'] == 'lose') & (df['Stake_Amount'] > df['Prev_Stake'])
chasing_count = df['Chasing_Loss'].sum()

# ==== d. Excessive Daily Bets (>10 per day) ====
bets_per_day = df.groupby(df['Date'].dt.date).size()
high_freq_days = bets_per_day[bets_per_day > 10]

# ==== e. Win/Loss Streaks ====
def longest_streak(series, result):
    streak = max_streak = 0
    for val in series:
        if val == result:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

longest_loss_streak = longest_streak(df['Result'], 'lose')
longest_win_streak = longest_streak(df['Result'], 'win')

# ==== f. Sudden Stake Jump > 100% ====
daily_stake_df = daily_stake.reset_index()
daily_stake_df.columns = ['Date', 'Stake']
daily_stake_df['Prev_Stake'] = daily_stake_df['Stake'].shift(1)

# Prevent division by zero or NaN
prev_stake_safe = daily_stake_df['Prev_Stake'].replace(0, np.nan)

# Compute safe stake jump
stake_jump = ((daily_stake_df['Stake'] - daily_stake_df['Prev_Stake']) / prev_stake_safe) * 100
stake_jump = stake_jump.replace([np.inf, -np.inf], np.nan).fillna(0)

daily_stake_df['Stake_Jump_%'] = stake_jump

# Filter rows where jump > 100%
sharp_jumps = daily_stake_df[daily_stake_df['Stake_Jump_%'] > 100]

# ==== g. Risk Score (Simple Composite) ====
# Ensure no NaN in night_bet_ratio
safe_night_ratio = night_bet_ratio if pd.notna(night_bet_ratio) else 0

risk_score = (
    len(risky_days) +
    (safe_night_ratio > 25) * 2 +
    (chasing_count > 3) * 2 +
    len(high_freq_days) +
    (longest_loss_streak >= 5) * 2 +
    len(sharp_jumps)
)

# ==== Final Summary Table ====
insights = {
    'Total Bets': len(df),
    'Total Risky Days (Stake > 10k)': len(risky_days),
    'Night Betting Ratio (%)': round(night_bet_ratio, 2),
    'Chasing Loss Count': int(chasing_count),
    'Days with >10 Bets': len(high_freq_days),
    'Longest Win Streak': longest_win_streak,
    'Longest Loss Streak': longest_loss_streak,
    'Days with >100% Stake Jump': len(sharp_jumps),
    'Responsible Gambling Risk Score (out of ~10)': risk_score
}

insight_df = pd.DataFrame.from_dict(insights, orient='index', columns=['Value'])

# ==== Export ====
os.makedirs("exports/data", exist_ok=True)
insight_df.to_csv("exports/data/responsible_gambling_insights.csv", index=True)

print("Responsible Gambling Insights Exported:")
print(insight_df)

