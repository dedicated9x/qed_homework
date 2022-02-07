from pandas_profiling import ProfileReport
from _solution.tasks.qed.preprocessing.process_df import get_special_df


def create_report(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(r"C:\temp\qed\reports\df_reduced.html")


def main():
    df = get_special_df()
    a = 2
    # create_report(df)

if __name__ == '__main__':
    main()



