%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identification part
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}[2001]
\ProvidesClass{thesis}[2003/11/30 UCD Thesis Class]

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Declaration of options
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execution of options
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Package loading
%%%%%%%%%%%%%%%%%%%%%%%%%%%

\LoadClass[12pt,letterpaper]{report}
\usepackage{epsfig}
\usepackage{theorem}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{url}
\usepackage{moreverb}

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Special commands for
% package and class files
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Special class file commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Horizotal Margins
%\setlength{\evensidemargin}{0in} % For two-sided printing
\setlength{\oddsidemargin}{0.875in}
\setlength{\textwidth}{5.625in}
% Vertical Margins
\setlength{\topmargin}{.5in}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0in}
% 1.5 inch top margin
\setlength{\textheight}{8in}
% 1.5 inch bottom margin (footer .75 in from bottom)
\setlength{\footskip}{0.75in}

\setlength{\parindent}{4ex}
\setlength{\arraycolsep}{0.25\arraycolsep}
\tolerance=500

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Effective \baselinestretch          %
% values for different font sizes.    %
% Latex Companion page 53             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% spacing          | 10pt  11pt  12pt %
%------------------|------------------%
% one and one-half | 1.25  1.21  1.24 %
% double           | 1.67  1.62  1.66 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set line spacing
\newcommand{\singlespace}{0.8}
\newcommand{\normalspace}{1.66}
\newcommand{\dblspace}{1.66}
\newcommand{\oneplushalf}{1.24}
\renewcommand{\baselinestretch}{\normalspace}
\newcommand{\@linespace}{%
  \renewcommand{\baselinestretch}{\singlespace}
  \normalsize
  \null\par
  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
}

% Setup counters for sections (add more as needed)
\@addtoreset{section}{chapter}
\@addtoreset{subsection}{section}
\@addtoreset{subsubsection}{subsection}
\setcounter{secnumdepth}{3}
\renewcommand{\thepart}{}
\renewcommand{\thechapter}{\arabic{chapter}}
\renewcommand{\thesection}{\thechapter.\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}

% Setup counters for figures and tables
\@addtoreset{figure}{chapter}
\@addtoreset{table}{chapter}
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}

\@addtoreset{equation}{chapter}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup Tables of Contents,
% List of Figures,
% List of Tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\l@part}{\@dottedtocline{0}{0em}{0em}}
\renewcommand{\l@chapter}{\@dottedtocline{0}{0em}{1.4em}}
\renewcommand{\l@section}{\@dottedtocline{1}{0em}{2.3em}}
\renewcommand{\l@subsection}{\@dottedtocline{2}{0em}{3.2em}}
\renewcommand{\l@subsubsection}{\@dottedtocline{2}{0em}{4.1em}}
\renewcommand{\l@figure}{\@dottedtocline{0}{0em}{2.4em}}
\renewcommand{\l@table}{\@dottedtocline{0}{0em}{2.4em}}
\renewcommand{\tableofcontents}{
  \clearpage
  \vspace*{0.25in}
  \centerheading{CONTENTS}
  \@starttoc{toc}
  \clearpage
}
\renewcommand{\listoffigures}{
  \clearpage
  \vspace*{0.25in}
  \part[Figures]{FIGURES}
  \addtocontents{lof}{Figure}
  \@starttoc{lof}
  \clearpage
}
\renewcommand{\listoftables}{
  \clearpage
  \vspace*{0.25in}
  \part[Tables]{TABLES}
  \addtocontents{lot}{Table}
  \@starttoc{lot}
  \addtocontents{toc}{\underline{Chapter}}
  \clearpage
}
\newcommand{\centerheading}[1]{\begin{center}{\textbf{#1}}\end{center}}

%%%%%%%%%%%%%%%%%%%%
% Front-matter
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define parts of title page.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\title}[1]{
  \gdef\@title{#1}
}
\renewcommand{\author}[2]{
  \gdef\@fwdname{#1 #2}
  \gdef\@revname{#2, #1}
}
\newcommand{\degree}[2]{
  \gdef\@fulldegree{#1}
  \gdef\@degree{#2}
}
\newcommand{\degreeyear}[1]{
  \gdef\@degreeyear{#1}
}
\newcommand{\otherdegrees}[1]{
  \gdef\@otherdegrees{#1}
}
\newcommand{\dept}[2]{
  \gdef\@deptorg{#1}
  \gdef\@deptname{#2}
}
\renewcommand{\titlepage}{
  \clearpage
  \renewcommand{\baselinestretch}{\dblspace}
  \normalsize

  \vspace*{1.25in}
  \begin{center}
    \uppercase\expandafter{\@title}\\
    by\\
    \@fwdname\\
    \@otherdegrees\\
  \end{center}
  \null\par
  \null\par
  \begin{center}
    A thesis submitted to the\\
    University of Colorado Denver \\
    in partial fulfillment\\
    of the requirements for the degree of\\
    \@fulldegree\\
    \@deptorg \@deptname\\
    \@degreeyear\\
  \end{center}

  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
  \clearpage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Environment for dedication or acknowledgment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{block}{%
  \renewcommand{\baselinestretch}{\oneplushalf}
  \normalsize
  \noindent
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Approval page
% Add more readers as necessary.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\advisor}[2]{
  \gdef\@advisor{#2}
  \gdef\@advisortitle{#1}
}
\newcommand{\readerone}[1]{
  \gdef\@readerone{#1}
}
\newcommand{\readertwo}[1]{
  \gdef\@readertwo{#1}
}
\newcommand{\readerthree}[1]{
  \gdef\@readerthree{#1}
}
\newcommand{\readerfour}[1]{
  \gdef\@readerfour{#1}
}
\newcommand{\@signature}[1]{
  \vspace{.40in}
  \renewcommand{\baselinestretch}{\singlespace}
  \normalsize
  \begin{center}
    \rule{3in}{0.4pt}\\
    #1
  \end{center}
  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
}
\newcommand{\@dateline}{%
  \renewcommand{\baselinestretch}{\singlespace}
  \normalsize

  \vspace{.5in}
  \vfill
  \begin{flushright}
    \parbox[]{1.5in}{\rule{1.5in}{0.4pt}\\\centerline{Date}}
  \end{flushright}
  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
}
\newcommand{\approvalpage}{
  \clearpage
  \vspace*{0.5in}
  \renewcommand{\baselinestretch}{\dblspace}
  \normalsize

  \begin{center}
    This thesis for the \@fulldegree\ \\
    degree by\\
    \@fwdname\\
    has been approved \\
    by\\
  \end{center}
  \@signature{\@advisor}
  \@signature{\@readerone}
  \@signature{\@readertwo}
  \@signature{\@readerthree}
  \@signature{\@readerfour}

  \@dateline

  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
  \clearpage
}

%%%%%%%%%%%%%%%%
% Abstract page
%%%%%%%%%%%%%%%%

\newcommand{\@abstractsignature}[1]{
  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
  \noindent
  This abstract accurately represents the content of the candidate's thesis. I
  recommend its publication.
  \null\par
  \null\par
  \begin{flushright}
    \renewcommand{\baselinestretch}{\singlespace}
    \normalsize
    Signed~
    \parbox[t]{0.5\linewidth}{
      \rule{\linewidth}{0.4pt}
      \centerline{#1}
    }
  \end{flushright}

  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
}
\newcommand{\@abstractheading}{
  \renewcommand{\baselinestretch}{\singlespace}
  \normalsize
  \vspace*{0.24in}
  \noindent
  \@revname\ (\@degree)
  \newline
  \@linespace
  \noindent
  \@title
  \@linespace
  \noindent
  Thesis directed by \@advisortitle\ \@advisor

  \renewcommand{\baselinestretch}{\normalspace}
  \normalsize
}
\newcommand{\abstractpage}[1]{
  \clearpage
  \pagestyle{empty}
  \vspace*{0.25in}
  \@abstractheading

  \renewcommand{\baselinestretch}{\oneplushalf}
  \normalsize
  \@linespace
  \@linespace
  \@linespace
  \@linespace

  \centerline{ABSTRACT}
  #1

  \@linespace
  \@linespace
  \@linespace
  \@linespace

  \@abstractsignature{\expandafter\@advisor}
  \clearpage
  \pagestyle{empty}
}

%%%%%%%%%%%%%%%%%%
% Section commands
%%%%%%%%%%%%%%%%%%
%
% The part section is used for List of Figures and Tables
% For some reason, the startsection method does not work
% for this level
%

\renewcommand{\part}{\secdef\@part\@spart}
\renewcommand{\@part}[2][]{%
  \centerheading{#2}
  \addcontentsline{toc}{part}{#1}
}
\renewcommand{\@spart}[1]{%
  \centerheading{#1}
}

\renewcommand{\chapter}{\clearpage
  \vspace*{0.24in}
  \@afterindenttrue
  \secdef\@chapter\@schapter
}
\renewcommand{\@chapter}[2][]{%
  \refstepcounter{chapter}
  \begin{flushleft}
  {\bf\thechapter.~~#2}
  \end{flushleft}
  \vspace{-1em}
  \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter.}#1}
  \@afterheading
}
\renewcommand{\@schapter}[1]{%
  \clearpage
  \begin{flushleft}
  {\bf#1}
  \end{flushleft}
  \vspace{-1em}
  \@afterheading
}

\renewcommand{\section}{
  \vspace{-0.24in}
  \vspace*{0.24in}
  \@afterindenttrue
  \secdef\@section\@ssection
}
\newcommand{\@section}[2][]{%
  \refstepcounter{section}
  \begin{flushleft}
  {\bf\thesection~~#2}
  \end{flushleft}
  \vspace{-1em}
  \addcontentsline{toc}{section}{\protect\numberline{\thesection}#1}
  \@afterheading
}
\newcommand{\@ssection}[1]{%
  \clearpage
  \begin{flushleft}
  {\bf#1}
  \end{flushleft}
  \vspace{-1em}
  \@afterheading
}

\renewcommand{\subsection}{
  \vspace{-0.24in}
  \vspace*{0.24in}
  \@afterindenttrue
  \secdef\@subsection\@ssubsection
}
\newcommand{\@subsection}[2][]{%
  \refstepcounter{subsection}
  \begin{flushleft}
  {\bf\thesubsection~~#2}
  \end{flushleft}
  \vspace{-1em}
  \addcontentsline{toc}{subsection}{\protect\numberline{\thesubsection}#1}
  \@afterheading
}
\newcommand{\@ssubsection}[1]{%
  \clearpage
  \begin{flushleft}
  {\bf#1}
  \end{flushleft}
  \vspace{-1em}
  \@afterheading
}

\renewcommand{\subsubsection}{
  \vspace{-0.24in}
  \vspace*{0.24in}
  \@afterindenttrue
  \secdef\@subsubsection\@ssubsubsection
}
\newcommand{\@subsubsection}[2][]{%
  \refstepcounter{subsubsection}
  \begin{flushleft}
  {\bf\thesubsubsection~~#2}
  \end{flushleft}
  \vspace{-1em}
  \addcontentsline{toc}{subsubsection}{\protect\numberline{\thesubsubsection}#1}
  \@afterheading
}
\newcommand{\@ssubsubsection}[1]{%
  \clearpage
  \begin{flushleft}
  {\bf#1}
  \end{flushleft}
  \vspace{-1em}
  \@afterheading
}


% New environment for appendix
\renewenvironment{appendix}{%
  \clearpage
  \addtocontents{toc}{\noindent\underline{Appendix}}

  \renewcommand{\chapter}{\clearpage
    \vspace*{0.24in}
    \@afterindenttrue
    \secdef\@chapter\@schapter
  }

  \renewcommand{\@chapter}[2][]{%
    \refstepcounter{chapter}
    \begin{flushleft}
    {\bf APPENDIX \thechapter.~~##2}
    \end{flushleft}
    \vspace{-1em}
    \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter.}##1}
    \@afterheading
  }

  \renewcommand{\@schapter}[1]{%
    \clearpage
    \begin{flushleft}
    {\bf##1}
    \end{flushleft}
    \vspace{-1em}
    \@afterheading
  }

  \renewcommand{\thechapter}{\Alph{chapter}}

  \setcounter{chapter}{0}}

  {\renewcommand{\thechapter}{\arabic{chapter}}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Special commands for formatting the
% captions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newlength{\C@aptionWidth}
\newcommand{\@maketblcaption}[2]{%
  \vspace{10pt}
  \settowidth{\C@aptionWidth}{{\bf Table \thetable:~~}#2}

    \renewcommand{\baselinestretch}{\singlespace}
    \normalsize

  \ifnum\C@aptionWidth<\linewidth
    \begin{center}
    {\bf Table \thetable:~~}#2
    \end{center}
  \else
    {\bf Table \thetable:~~}#2
  \fi
}
\newcommand{\@makefigcaption}[2]{%
  \vspace{10pt}
  \settowidth{\C@aptionWidth}{{\bf Figure \thefigure:~~}#2}

    \renewcommand{\baselinestretch}{\singlespace}
    \normalsize

  \ifnum\C@aptionWidth<\linewidth
    \begin{center}
    {\bf Figure \thefigure:~~}#2
    \end{center}
  \else
    {\bf Figure \thefigure:~~}#2
  \fi
}
\renewcommand{\figure}{\let\@makecaption\@makefigcaption\@float{figure}}
\renewcommand{\table}{\let\@makecaption\@maketblcaption\@float{table}}


% Places Reference page in table of contents
\renewcommand{\thebibliography}[1]{%
  \renewcommand{\baselinestretch}{1}
  \normalsize

  \section*{\vspace*{.25in}~\centerheading{REFERENCES}
            \markboth{REFERENCES}{REFERENCES}}
  \addcontentsline{toc}{part}{\underline{References}}
  \list
  {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
  \advance\leftmargin\labelsep
  \usecounter{enumi}}
  \def\newblock{\hskip .11em plus .33em minus -.07em}
  \sloppy
  \sfcode`\.=1000\relax}

% FOOTNOTES
% change it so that footnotes are printed in single spacing
%
\long\def\@footnotetext#1{\insert\footins{\renewcommand\baselinestretch{1}
    \footnotesize
    \interlinepenalty\interfootnotelinepenalty 
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
   \edef\@currentlabel{\csname p@footnote\endcsname\@thefnmark}\@makefntext
    {\rule{\z@}{\footnotesep}\ignorespaces
      #1\strut}\renewcommand\baselinestretch{\@spacing}}}
