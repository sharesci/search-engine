export interface ISearchResults {
    errno: number,
    results: [{
        _id: string,
        authors: [{keyname: string, forenames: string}],
        title: string,
        score: number
    }],
    numResults: number
}