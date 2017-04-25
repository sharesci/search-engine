export interface ISearchResults {
    errno: number,
    results: [{
        _id: string,
        title: string,
        score: number
    }]
}